from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .application import create_agent
from .config import Settings
from .data.ingest import Ingestor
from .data.split import DatasetSplit
from .domain import DataMateError, Question
from .models.registry import MODELS, inventory


class ChatRequest(BaseModel):
    preset: str = "demo"
    split: str = "train"
    business_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=4000)
    as_of_date: str


def create_app(workspace: str | Path = ".") -> FastAPI:
    workspace = Path(workspace).resolve()
    app = FastAPI(title="DataMate Agent Lab", version="0.1.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"])
    assets = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=assets), name="assets")
    state = {"agent": None, "identity": None, "lock": threading.Lock()}
    splits = {"train": workspace / "train/train", "public_test": workspace / "public_test/public_test"}

    def split_path(name):
        if name not in splits or not splits[name].is_dir():
            raise HTTPException(400, "Split không có trong workspace")
        return splits[name]

    def settings_for(preset, split):
        if not preset or Path(preset).name != preset or preset not in presets():
            raise HTTPException(400, "Preset không hợp lệ")
        settings = Settings.load(workspace / "configs" / f"{preset}.json")
        settings.split_root = str(split_path(split))
        for field in ("model_dir", "assistant_dir", "cache_dir", "result_dir"):
            value = getattr(settings, field)
            if value is not None:
                path = (workspace / value).resolve()
                if not path.is_relative_to(workspace):
                    raise HTTPException(400, "Preset web phải dùng đường dẫn bên trong workspace")
                setattr(settings, field, str(path))
        return settings.validate()

    def presets():
        found = {}
        for path in sorted((workspace / "configs").glob("*.json")):
            if path.stem.startswith("submission"):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("backend") in {"demo", "transformers"}:
                    found[path.stem] = raw
            except (ValueError, OSError):
                continue
        return found

    @app.middleware("http")
    async def headers(request, call_next):
        # Requests from another local website cannot invoke expensive inference.
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin not in {f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Origin không được phép"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/api/state")
    def get_state():
        gpu = "Chưa xác định GPU"
        try:
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            outcome = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, **kwargs)
            if outcome.returncode == 0:
                gpu = outcome.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        available = []
        for name, raw in presets().items():
            main = workspace / raw.get("model_dir", "models/none") / "artifact_manifest.json"
            ready = raw["backend"] == "demo" or main.exists()
            if raw.get("assistant_dir"):
                ready = ready and (workspace / raw["assistant_dir"] / "artifact_manifest.json").exists()
            available.append({"key": name, "model_key": raw.get("model_key"), "mode": raw.get("mode", "single"),
                              "backend": raw["backend"], "quantization": raw.get("quantization", "none"), "local_files_present": ready})
        return {"gpu": gpu, "presets": available, "splits": [name for name, path in splits.items() if path.is_dir()],
                "models": [spec.to_dict() for spec in MODELS.values()], "busy": state["lock"].locked(),
                "notice": "Demo luôn abstain. Cần tải và xác minh checkpoint trước khi chạy LLM thật."}

    @app.get("/api/businesses")
    def businesses(split: str = "train"):
        try:
            return {"businesses": DatasetSplit(split_path(split)).businesses()}
        except DataMateError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/schema")
    def schema(business: str, split: str = "train"):
        try:
            dataset = DatasetSplit(split_path(split))
            return {"catalog": Ingestor(dataset, workspace / ".cache/datamate").catalog(business).summary(),
                    "readme": dataset.readme(business), "rulebook_sha256": dataset.rulebook_hash}
        except DataMateError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        if not state["lock"].acquire(blocking=False):
            raise HTTPException(409, "Agent đang chạy một câu khác; chờ câu đó hoàn tất")
        try:
            settings = settings_for(request.preset, request.split)
            identity = settings.to_dict()
            if state["identity"] != identity:
                state["agent"] = None
                state["identity"] = None
                gc.collect()
                if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                    sys.modules["torch"].cuda.empty_cache()
                state["agent"] = create_agent(settings)
                state["identity"] = identity
            question = Question("web_" + uuid.uuid4().hex[:16], request.business_id, request.question, request.as_of_date)
            run = state["agent"].run(question)
            return {**run.to_dict(question.id), "backend": settings.backend, "mode": settings.mode,
                    "loaded_parameters": 0 if settings.backend == "demo" else inventory(settings.model_key, settings.assistant_key)["total_parameters"],
                    "inventory": inventory(settings.model_key, settings.assistant_key)}
        except HTTPException:
            raise
        except DataMateError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"Không chạy được model: {type(exc).__name__}: {str(exc)[:500]}") from exc
        finally:
            state["lock"].release()

    return app
