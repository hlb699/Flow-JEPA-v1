#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
cd "$REPO_DIR"

./train_tworoom_paper_4gpu.sh \
    2>&1 | tee logs/train_tworoom_paper_4gpu.log

CUDA_VISIBLE_DEVICES=0 ./eval_tworoom_paper_4gpu.sh
