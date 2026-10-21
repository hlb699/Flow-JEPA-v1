#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/data/private/Flow-JEPA"
RUN_NAME="flow-jepa-pusht-paper-1gpu-strict"
CHECKPOINT="$REPO_DIR/stable-wm/checkpoints/$RUN_NAME/weights_epoch_20.pt"
RESULT_DIR="$REPO_DIR/logs/eval_results/${RUN_NAME}_cem10_clean"

export STABLEWM_HOME="$REPO_DIR/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1

while pgrep -f "[t]rain.py --config-name=pusht_paper.*subdir=$RUN_NAME" >/dev/null; do
    sleep 60
done

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Training exited before $CHECKPOINT was created." >&2
    exit 1
fi

cd "$REPO_DIR"
exec .venv/bin/python eval.py --config-name=pusht \
    policy="$CHECKPOINT" \
    seed=42 \
    eval.num_eval=50 \
    solver.n_steps=10 \
    output.save_video=false \
    output.dir="$RESULT_DIR" \
    output.filename=results.txt
