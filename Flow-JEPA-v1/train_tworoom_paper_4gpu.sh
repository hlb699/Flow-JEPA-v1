#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
export STABLEWM_HOME="$REPO_DIR/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
exec .venv/bin/python train.py --config-name=tworoom_paper_4gpu
