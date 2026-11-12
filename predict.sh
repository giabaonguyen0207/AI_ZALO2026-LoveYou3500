#!/bin/sh
set -eu
scratch_root="${DATAMATE_SCRATCH_DIR:-/work}"
result_root="${DATAMATE_RESULT_DIR:-/result}"
export TMPDIR="${TMPDIR:-$scratch_root/tmp}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$scratch_root/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$scratch_root/vllm-config}"
mkdir -p "$result_root/logs" "$scratch_root" "$TMPDIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
# Hard wall-clock watchdog includes Python imports, artifact verification, model
# loading, ingestion and generation. Already flushed answers survive termination.
exec timeout --signal=TERM --kill-after="${DATAMATE_KILL_AFTER_SECONDS:-15}s" \
    "${DATAMATE_WATCHDOG_SECONDS:-21570}s" \
    "${DATAMATE_RUNTIME_PYTHON:-python}" -m datamate run \
    --config "${DATAMATE_CONFIG:-/app/configs/submission.json}" "$@"
