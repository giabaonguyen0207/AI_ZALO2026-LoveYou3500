#!/bin/sh
set -eu
mkdir -p /result/logs /work
# Hard wall-clock watchdog includes Python imports, artifact verification, model
# loading, ingestion and generation. Already flushed answers survive termination.
exec timeout --signal=TERM --kill-after=15s 21570s \
    python -m datamate run --config "${DATAMATE_CONFIG:-/app/configs/submission.json}" "$@"
