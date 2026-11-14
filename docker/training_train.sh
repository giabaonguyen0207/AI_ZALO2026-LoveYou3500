#!/bin/bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
BASE="${1:?Usage: train.sh VERIFIED_PUBLIC_BASE NEW_WORK_DIR}"
WORK="${2:?Provide NEW_WORK_DIR outside training_code}"
ENV_DIR="$WORK/environment"
test ! -e "$WORK" || { echo 'Choose a new reproduction work directory' >&2; exit 2; }
mkdir -p "$WORK"
python -m venv --without-pip "$ENV_DIR"
python -m pip --python "$ENV_DIR/bin/python" install --no-index --require-hashes \
  --find-links "$HERE/wheelhouse" -r "$HERE/wheelhouse/requirements.resolved.txt"
python -m pip --python "$ENV_DIR/bin/python" check
for stage in stage1 stage2; do
  "$ENV_DIR/bin/python" "$HERE/reproduce.py" --evidence "$HERE" \
    --base "$BASE" --work "$WORK" --stage "$stage"
done
