#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STABLEWM_HOME="$repo_dir/stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"

checkpoint="$STABLEWM_HOME/checkpoints/flow-jepa-pusht/weights_epoch_20.pt"

while [[ ! -f "$checkpoint" ]]; do
    if ! pgrep -f '[.]venv/bin/python train.py data=pusht' >/dev/null; then
        echo "Training exited before the epoch-20 checkpoint was created." >&2
        exit 1
    fi
    sleep 60
done

while pgrep -f '[.]venv/bin/python train.py data=pusht' >/dev/null; do
    sleep 60
done

cd "$repo_dir"
exec .venv/bin/python eval.py --config-name=pusht \
    policy="$checkpoint"
