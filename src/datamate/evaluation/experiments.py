from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..data.split import read_jsonl
from ..domain import DataMateError, json_text
from .scoring import score


def compare(configs, ids, references, output):
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise DataMateError("Compare output cần thư mục mới để không trộn kết quả")
    destination.mkdir(parents=True, exist_ok=True)
    question_ids = set(json.loads(Path(ids).read_text(encoding="utf-8")))
    refs = read_jsonl(Path(references))
    rows = []
    for i, path in enumerate(configs):
        result_dir = destination / f"{i:02d}_{Path(path).stem}"
        # A fresh process releases the previous model's CUDA memory. Every
        # variant uses exactly the same held-out question IDs and data split.
        subprocess.run([sys.executable, "-m", "datamate", "run", "--config", str(path), "--ids", str(ids), "--result-dir", str(result_dir)], check=True)
        report = score(read_jsonl(result_dir / "submission.jsonl"), refs, question_ids)
        summary = json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8"))
        logs = [json.loads(p.read_text(encoding="utf-8")) for p in (result_dir / "logs").glob("*.json")]
        latency = sorted(record["elapsed_seconds"] for record in logs)
        rows.append({"config": str(path), **{key: value for key, value in report.items() if key != "details"},
                     "wall_time_seconds": summary["wall_time_seconds"], "load_seconds": summary["load_seconds"],
                     "latency_p50_seconds": latency[len(latency)//2] if latency else None,
                     "latency_p95_seconds": latency[min(len(latency)-1, int(len(latency)*0.95))] if latency else None})
        (destination / "comparison.json").write_text(json_text(rows, indent=2), encoding="utf-8")
    return rows
