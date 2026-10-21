#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
RUN_NAME="flow-jepa-pusht-paper-1gpu-strict"

export STABLEWM_HOME="$REPO_DIR/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
exec .venv/bin/python train.py --config-name=pusht_paper \
    output_model_name="$RUN_NAME" \
    subdir="$RUN_NAME"
