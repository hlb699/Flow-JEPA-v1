#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

export STABLEWM_HOME="$repo_dir/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

checkpoint="$STABLEWM_HOME/checkpoints/flow-jepa-pusht/weights_epoch_12.pt"

.venv/bin/python train.py \
    data=pusht \
    output_model_name=flow-jepa-pusht \
    loader.prefetch_factor=1 \
    trainer.max_epochs=8 \
    model_initialization.enabled=true \
    model_initialization.checkpoint="$checkpoint" \
    epoch_offset=12 \
    2>&1 | tee logs/train_pusht_resume12.log

.venv/bin/python eval.py --config-name=pusht \
    policy="$STABLEWM_HOME/checkpoints/flow-jepa-pusht/weights_epoch_20.pt" \
    2>&1 | tee logs/eval_pusht.log
