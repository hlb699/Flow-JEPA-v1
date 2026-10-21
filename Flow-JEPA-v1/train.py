import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import (
    RankAwareTrainingSeedCallback,
    SaveCkptCallback,
    get_column_normalizer,
    get_img_preprocessor,
)


def convert_legacy_vit_state_dict(state_dict):
    """Convert pre-Transformers-5 ViT parameter names to current names."""
    replacements = (
        ("encoder.layer.", "layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )
    converted = {}
    for key, value in state_dict.items():
        converted_key = key
        for old, new in replacements:
            converted_key = converted_key.replace(old, new)
        if converted_key in converted:
            raise ValueError(
                f"ViT checkpoint key conversion collision: {converted_key}"
            )
        converted[converted_key] = value
    return converted


def initialize_model_from_checkpoint(model, checkpoint, *, strict=True):
    """Warm-start all model parameters from save_pretrained weights."""
    checkpoint = Path(str(checkpoint)).expanduser()
    if checkpoint.is_dir():
        weight_files = sorted(checkpoint.glob("*.pt"))
        if len(weight_files) != 1:
            raise ValueError(
                "model initialization folders must contain exactly one .pt "
                "file; provide an explicit file when multiple checkpoints "
                "are present"
            )
        checkpoint = weight_files[0]
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise ValueError(
            "model initialization checkpoint must be an existing .pt file "
            "or a folder containing one .pt file"
        )

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=bool(strict))
    return model


def initialize_visual_encoder_from_checkpoint(
    model,
    checkpoint,
    *,
    load_projector=True,
    strict=True,
):
    """Load the visual latent coordinate system from a pretrained model."""
    checkpoint = Path(str(checkpoint)).expanduser()
    if checkpoint.exists():
        if checkpoint.is_dir():
            weight_files = sorted(checkpoint.glob("*.pt"))
            if len(weight_files) != 1:
                raise ValueError(
                    "pretrained visual checkpoint folders must contain "
                    "exactly one .pt file; provide an explicit file when "
                    "multiple checkpoints are present"
                )
            checkpoint = weight_files[0]
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError(
                "pretrained visual checkpoint must be a .pt file or a "
                "folder containing one .pt file"
            )
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        def component_state(name):
            prefixes = (f"{name}.", f"model.{name}.")
            for prefix in prefixes:
                selected = {
                    key[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                if selected:
                    return selected
            raise KeyError(
                f"checkpoint contains no parameters for {name}"
            )

        encoder_state = component_state("encoder")
        if any(key.startswith("encoder.layer.") for key in encoder_state):
            encoder_state = convert_legacy_vit_state_dict(encoder_state)
        model.encoder.load_state_dict(encoder_state, strict=bool(strict))
        if load_projector:
            model.projector.load_state_dict(
                component_state("projector"),
                strict=bool(strict),
            )
    else:
        pretrained = swm.wm.utils.load_pretrained(str(checkpoint))
        model.encoder.load_state_dict(
            pretrained.encoder.state_dict(),
            strict=bool(strict),
        )
        if load_projector:
            model.projector.load_state_dict(
                pretrained.projector.state_dict(),
                strict=bool(strict),
            )
    return model


def flow_jepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, K, D)
    act_emb = output["act_emb"]

    # Loss
    ctx_emb = emb[:, :ctx_len]
    future_act = act_emb[:, ctx_len - 1 : ctx_len - 1 + n_preds]
    tgt_emb = emb[:, ctx_len : ctx_len + n_preds]
    output["pred_loss"] = self.model.flow_loss(ctx_emb, future_act, tgt_emb)
    sigreg_emb = emb.permute(1, 2, 0, 3).reshape(
        emb.size(1) * emb.size(2), emb.size(0), emb.size(3)
    )
    output["sigreg_loss"]= self.sigreg(sigreg_emb)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="fjepa")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    expected_global_batch_size = cfg.get(
        "expected_global_batch_size",
        None,
    )
    if expected_global_batch_size is not None:
        devices = cfg.trainer.get("devices", 1)
        num_nodes = int(cfg.trainer.get("num_nodes", 1))
        accumulate = int(
            cfg.trainer.get("accumulate_grad_batches", 1)
        )
        if not isinstance(devices, int):
            raise ValueError(
                "paper configs require trainer.devices to be an integer so "
                "the global batch size can be verified"
            )
        if accumulate != 1:
            raise ValueError(
                "paper configs require accumulate_grad_batches=1 because "
                "SIGReg is defined on each global data batch"
            )
        global_batch_size = int(cfg.loader.batch_size) * devices * num_nodes
        if global_batch_size != int(expected_global_batch_size):
            raise ValueError(
                "global batch size mismatch: "
                f"{cfg.loader.batch_size} per device x {devices} devices x "
                f"{num_nodes} nodes = {global_batch_size}, expected "
                f"{expected_global_batch_size}"
            )

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

        pretrained_visual_cfg = cfg.get("pretrained_visual_encoder", {})
        model_initialization_cfg = cfg.get("model_initialization", {})
        pretrained_visual_enabled = bool(
            pretrained_visual_cfg.get("enabled", False)
        )
        model_initialization_enabled = bool(
            model_initialization_cfg.get("enabled", False)
        )
        if pretrained_visual_enabled and model_initialization_enabled:
            raise ValueError(
                "pretrained_visual_encoder and model_initialization cannot "
                "both be enabled"
            )
        if pretrained_visual_enabled:
            checkpoint = pretrained_visual_cfg.get("checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "pretrained_visual_encoder.checkpoint is required "
                    "when pretrained visual initialization is enabled"
                )
            freeze_visual = bool(
                pretrained_visual_cfg.get("freeze", True)
            )
            cfg.model.freeze_encoder = freeze_visual
            cfg.model.freeze_projector = (
                freeze_visual
                and bool(
                    pretrained_visual_cfg.get(
                        "load_projector",
                        True,
                    )
                )
            )
        if model_initialization_enabled and not model_initialization_cfg.get(
            "checkpoint", None
        ):
            raise ValueError(
                "model_initialization.checkpoint is required when complete "
                "model initialization is enabled"
            )

        epoch_offset = int(cfg.get("epoch_offset", 0))
        if epoch_offset < 0:
            raise ValueError("epoch_offset must be non-negative")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    pretrained_visual_cfg = cfg.get("pretrained_visual_encoder", {})
    if bool(pretrained_visual_cfg.get("enabled", False)):
        initialize_visual_encoder_from_checkpoint(
            world_model,
            pretrained_visual_cfg.checkpoint,
            load_projector=bool(
                pretrained_visual_cfg.get("load_projector", True)
            ),
            strict=bool(pretrained_visual_cfg.get("strict", True)),
        )
        world_model.set_visual_encoder_frozen(
            bool(cfg.model.freeze_encoder),
            bool(cfg.model.freeze_projector),
        )
    model_initialization_cfg = cfg.get("model_initialization", {})
    if bool(model_initialization_cfg.get("enabled", False)):
        initialize_model_from_checkpoint(
            world_model,
            model_initialization_cfg.checkpoint,
            strict=bool(model_initialization_cfg.get("strict", True)),
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(flow_jepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
        epoch_offset=int(cfg.get("epoch_offset", 0)),
    )
    rank_seed_callback = RankAwareTrainingSeedCallback(cfg.seed)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[rank_seed_callback, object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=int(cfg.seed),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
