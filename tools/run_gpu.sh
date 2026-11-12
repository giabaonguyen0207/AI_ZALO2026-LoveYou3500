#!/usr/bin/env bash
# Use an already prepared Linux GPU environment. Installs/downloads nothing.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo 'Usage: bash tools/run_gpu.sh CONFIG SPLIT_ROOT NEW_RESULT_DIR [IDS_JSON]' >&2
    exit 2
fi
config=$1
split_root=$2
result_dir=$3
python_bin=${PYTHON_BIN:-python}
if [ -e "$result_dir" ]; then
    echo 'Use a new result directory for each run; this script does not resume.' >&2
    exit 2
fi
if [ ! -f "$config" ] || [ ! -f "$split_root/questions.jsonl" ] || [ ! -f "$split_root/rulebook.md" ] || [ ! -f "$split_root/calendar.csv" ] || [ ! -d "$split_root/data" ]; then
    echo 'Missing config or split files. Pass the inner split directory.' >&2
    exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$result_dir"
"$python_bin" -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in distributions())))' > "$result_dir/environment.txt"
nvidia-smi > "$result_dir/nvidia-smi.txt"
git rev-parse HEAD > "$result_dir/source-commit.txt"
git diff HEAD --binary > "$result_dir/source-working-tree.patch"
git status --porcelain > "$result_dir/source-status.txt"
git ls-files --others --exclude-standard > "$result_dir/untracked-files.txt"
cp "$config" "$result_dir/config.input.json"
extra=()
if [ "$#" -eq 4 ]; then
    cp "$4" "$result_dir/ids.input.json"
    extra=(--ids "$4")
fi
"$python_bin" -m datamate run --config "$config" --split-root "$split_root" --result-dir "$result_dir" "${extra[@]}" 2>&1 | tee "$result_dir/console.log"
