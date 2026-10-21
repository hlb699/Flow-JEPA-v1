#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
CHECKPOINT="$REPO_DIR/stable-wm/checkpoints/flow-jepa-pusht-paper-4gpu/weights_epoch_20.pt"

export STABLEWM_HOME="$REPO_DIR/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="3"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

.venv/bin/python eval.py --config-name=pusht \
    policy="$CHECKPOINT" \
    eval.num_eval=50 \
    solver.n_steps=10 \
    output.save_video=false \
    output.dir="$REPO_DIR/logs/eval_results/paper_4gpu_cem10_clean" \
    output.filename=results.txt \
    > logs/eval_pusht_paper_4gpu_cem10_clean.log 2>&1

.venv/bin/python eval.py --config-name=pusht \
    policy="$CHECKPOINT" \
    eval.num_eval=50 \
    solver.n_steps=10 \
    perturbation.enabled=true \
    perturbation.gaussian_noise.std=100 \
    perturbation.gaussian_noise.radius=10 \
    output.save_video=false \
    output.dir="$REPO_DIR/logs/eval_results/paper_4gpu_cem10_noisy_std100_r10" \
    output.filename=results.txt \
    > logs/eval_pusht_paper_4gpu_cem10_noisy_std100_r10.log 2>&1
