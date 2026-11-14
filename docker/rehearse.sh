#!/bin/bash
set -euo pipefail
IMAGE="${1:?Usage: rehearse.sh IMAGE ABSOLUTE_SPLIT_ROOT NEW_OUTPUT_ROOT}"
DATA="$(realpath -- "${2:?Provide input split}")"
OUT="$(realpath -m -- "${3:?Provide new output root}")"
test ! -e "$OUT" || { echo 'Choose a new output root' >&2; exit 2; }
mkdir -p "$OUT"
docker image inspect "$IMAGE" > "$OUT/image-inspect.json"
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
  --entrypoint bash "$IMAGE" -c \
  'set -eu; nvidia-smi; python -m pip check; ninja --version; nvcc --version; command -v g++; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); assert torch.cuda.is_available() and torch.cuda.device_count()==2; print([(torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) for i in range(2)])"' \
  > "$OUT/preflight.log" 2>&1
# Both containers start with their own cache. No host cache, network, host IPC,
# privileged mode, Docker socket or custom scoring arguments are supplied.
for attempt in 1 2; do
  mkdir -p "$OUT/run-$attempt"
  docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
    --entrypoint bash \
    --mount "type=bind,source=$DATA,target=/data,readonly" \
    --mount "type=bind,source=$OUT/run-$attempt,target=/result" \
    "$IMAGE" /code/predict.sh 2>&1 | tee "$OUT/run-$attempt.log"
done
echo "Outputs: $OUT. Verify both with tools/verify_submission.py and compare normalized answers before export."
