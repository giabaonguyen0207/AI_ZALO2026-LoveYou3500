from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .domain import DataMateError, Question, json_text


def add_config(parser):
    parser.add_argument("--config")
    parser.add_argument("--split-root")
    parser.add_argument("--result-dir")
    parser.add_argument("--backend", choices=["demo", "transformers"])


def load_settings(args):
    return Settings.load(getattr(args, "config", None), **{key: getattr(args, key, None) for key in ("split_root", "result_dir", "backend")})


def parser():
    app = argparse.ArgumentParser(prog="datamate", description="DataMate offline modular monolith")
    commands = app.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Đọc schema dữ liệu thực, không tải LLM")
    inspect.add_argument("--split-root", required=True)
    inspect.add_argument("--business", required=True)
    inspect.add_argument("--cache-dir", default=".cache/datamate")
    inspect.add_argument("--output")
    run = commands.add_parser("run", help="Chạy batch và ghi submission/logs từng câu")
    add_config(run)
    run.add_argument("--ids", help="File JSON array question IDs, thường lấy từ holdout train")
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    ask = commands.add_parser("ask", help="Hỏi một câu")
    add_config(ask)
    ask.add_argument("--business", required=True)
    ask.add_argument("--as-of", required=True)
    ask.add_argument("--question", required=True)
    stage = commands.add_parser("stage-model", help="Chuẩn bị ONLINE: tải đúng checkpoint pin, seal hash")
    stage.add_argument("--model", required=True)
    stage.add_argument("--output", required=True)
    commands.add_parser("models", help="Inventory các checkpoint đã xác minh metadata")
    serve = commands.add_parser("serve", help="Web local trên 127.0.0.1")
    serve.add_argument("--workspace", default=".")
    serve.add_argument("--port", type=int, default=7860)
    score = commands.add_parser("score", help="Chấm cục bộ với train references")
    score.add_argument("--predictions", required=True)
    score.add_argument("--references", required=True)
    score.add_argument("--ids")
    score.add_argument("--output")
    prepare = commands.add_parser("prepare-training", help="Chuẩn bị train/validation theo doanh nghiệp")
    prepare.add_argument("--split-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--kind", choices=["format", "trajectories"], default="format")
    prepare.add_argument("--logs")
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--sources", default="configs/training_sources.json")
    train = commands.add_parser("train", help="LoRA/QLoRA từ local checkpoint; Colab GPU")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    merge = commands.add_parser("merge", help="Merge adapter vào base và seal model offline")
    merge.add_argument("--model", required=True)
    merge.add_argument("--base", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)
    bench = commands.add_parser("compare", help="Chạy từng cấu hình độc lập trên cùng holdout")
    bench.add_argument("--configs", nargs="+", required=True)
    bench.add_argument("--ids", required=True)
    bench.add_argument("--references", required=True)
    bench.add_argument("--output", required=True)
    return app


def main():
    args = parser().parse_args()
    try:
        if args.command == "inspect":
            from .data.split import DatasetSplit, ensure_outside_dataset
            from .data.ingest import Ingestor
            result = Ingestor(DatasetSplit(args.split_root), args.cache_dir).catalog(args.business).summary()
            if args.output:
                output = ensure_outside_dataset(args.output, args.split_root)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json_text(result, indent=2), encoding="utf-8")
        elif args.command == "models":
            from .models.registry import MODELS
            result = [spec.to_dict() for spec in MODELS.values()]
        elif args.command == "stage-model":
            from .models.artifacts import stage_model
            result = stage_model(args.model, args.output)
            result = {"output": args.output, "source": result["source"], "files_verified": len(result["files"])}
        elif args.command == "run":
            from .application import run_batch
            ids = set(json.loads(Path(args.ids).read_text(encoding="utf-8"))) if args.ids else None
            result = run_batch(load_settings(args), ids, args.limit, args.resume)
        elif args.command == "ask":
            from .application import create_agent
            question = Question("interactive", args.business, args.question, args.as_of)
            result = create_agent(load_settings(args)).run(question).to_dict(question.id)
        elif args.command == "serve":
            from .api import create_app
            import uvicorn
            uvicorn.run(create_app(args.workspace), host="127.0.0.1", port=args.port)
            return
        elif args.command == "score":
            from .data.split import read_jsonl
            from .evaluation.scoring import score
            ids = set(json.loads(Path(args.ids).read_text(encoding="utf-8"))) if args.ids else None
            result = score(read_jsonl(Path(args.predictions)), read_jsonl(Path(args.references)), ids)
            if args.output:
                Path(args.output).write_text(json_text(result, indent=2), encoding="utf-8")
            result = {key: value for key, value in result.items() if key != "details"}
        elif args.command == "prepare-training":
            from .training.dataset import prepare_training
            result = prepare_training(args.split_root, args.output, args.kind, args.logs, args.validation_fraction, args.seed, args.sources)
        elif args.command == "train":
            from .training.finetune import train
            result = train(args.config, args.resume)
        elif args.command == "merge":
            from .training.finetune import merge
            result = merge(args.model, args.base, args.adapter, args.output)
        elif args.command == "compare":
            from .evaluation.experiments import compare
            result = compare(args.configs, args.ids, args.references, args.output)
        print(json_text(result, indent=2))
    except (DataMateError, FileNotFoundError, ValueError) as exc:
        print(f"DataMate: {exc}", file=sys.stderr)
        raise SystemExit(2)
