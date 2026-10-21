#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
RUN_NAME="flow-jepa-tworoom-paper-4gpu"
CHECKPOINT="$REPO_DIR/stable-wm/checkpoints/$RUN_NAME/weights_epoch_20.pt"

export STABLEWM_HOME="$REPO_DIR/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi

cd "$REPO_DIR"

.venv/bin/python eval.py --config-name=tworoom \
    policy="$CHECKPOINT" \
    seed=42 \
    eval.num_eval=50 \
    solver.n_steps=30 \
    output.save_video=false \
    output.dir="$REPO_DIR/logs/eval_results/${RUN_NAME}_cem30_clean" \
    output.filename=results.txt \
    > "logs/eval_${RUN_NAME}_cem30_clean.log" 2>&1

.venv/bin/python eval.py --config-name=tworoom \
    policy="$CHECKPOINT" \
    seed=42 \
    eval.num_eval=50 \
    solver.n_steps=30 \
    perturbation.enabled=true \
    perturbation.gaussian_noise.std=100 \
    perturbation.gaussian_noise.radius=35 \
    output.save_video=false \
    output.dir="$REPO_DIR/logs/eval_results/${RUN_NAME}_cem30_noisy_std100_r35" \
    output.filename=results.txt \
    > "logs/eval_${RUN_NAME}_cem30_noisy_std100_r35.log" 2>&1
