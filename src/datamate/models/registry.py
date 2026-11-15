from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..domain import DataMateError

CUTOFF = datetime(2026, 10, 1, tzinfo=timezone.utc)
PARAMETER_CAP = 15_000_000_000


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    parameters: int
    revision_date: str
    license: str = "apache-2.0"
    assistant_for: str | None = None

    def to_dict(self):
        return asdict(self)


MODELS = {
    "qwen35-9b": ModelSpec("qwen35-9b", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 9_653_104_368, "2026-03-02T00:51:43Z"),
    "qwen35-upscaled14b-v2": ModelSpec("qwen35-upscaled14b-v2", "Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft", "f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2", 14_599_485_040, "2026-03-24T11:13:01Z"),
    "gemma4-12b": ModelSpec("gemma4-12b", "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", 11_959_730_224, "2026-07-20T16:42:08Z"),
    "gemma4-12b-assistant": ModelSpec("gemma4-12b-assistant", "google/gemma-4-12B-it-assistant", "46d4c6f13f0ac0ad827b915669b8df9b81c64c51", 422_856_964, "2026-08-20T21:49:20Z", assistant_for="gemma4-12b"),
    "vietnamese-embedding": ModelSpec("vietnamese-embedding", "AITeamVN/Vietnamese_Embedding", "dea33aa1ab339f38d66ae0a40e6c40e0a9249568", 567_754_752, "2025-08-25T13:32:21Z"),
    "vietnamese-reranker": ModelSpec("vietnamese-reranker", "AITeamVN/Vietnamese_Reranker", "f536976248403314225d7fdfdbc87f0e9516a54e", 567_755_777, "2025-05-06T07:15:35Z"),
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise DataMateError(f"Model không có trong registry: {key}")
    spec = MODELS[key]
    if len(spec.revision) != 40 or datetime.fromisoformat(spec.revision_date.replace("Z", "+00:00")) >= CUTOFF:
        raise DataMateError("Checkpoint chưa pin hoặc không trước cutoff")
    return spec


def inventory(main_key: str, assistant_key: str | None = None, *, extra_keys=()) -> dict:
    main = get_model(main_key)
    specs = [main]
    if assistant_key:
        assistant = get_model(assistant_key)
        if assistant.assistant_for != main.key:
            raise DataMateError("Assistant không được nhà phát hành ghép với model chính này")
        specs.append(assistant)
    specs.extend(get_model(key) for key in extra_keys)
    specs = list({spec.key: spec for spec in specs}.values())
    total = sum(spec.parameters for spec in {spec.key: spec for spec in specs}.values())
    if total > PARAMETER_CAP:
        raise DataMateError(f"Tổng {total:,} tham số vượt giới hạn 15B")
    return {"models": [spec.to_dict() for spec in specs], "total_parameters": total,
            "limit": PARAMETER_CAP, "quantization_changes_count": False}


def inventory_from_settings(settings) -> dict:
    extras = []
    if settings.rag.mode in {'dense', 'dense_rerank'}:
        extras.append(settings.rag.embedding_key)
    if settings.rag.mode == 'dense_rerank':
        extras.append(settings.rag.reranker_key)
    return inventory(settings.model_key, settings.assistant_key, extra_keys=extras)


def model_paths(settings) -> dict:
    """Every active checkpoint, using the same roles as full inventory."""
    result = {settings.model_key: settings.model_dir}
    if settings.assistant_key:
        result[settings.assistant_key] = settings.assistant_dir
    if settings.rag.mode != 'none':
        result[settings.rag.embedding_key] = settings.rag.embedding_dir
    if settings.rag.mode == 'dense_rerank':
        result[settings.rag.reranker_key] = settings.rag.reranker_dir
    return result


def relocate_models(settings, paths):
    settings.model_dir = str(paths[settings.model_key])
    if settings.assistant_key:
        settings.assistant_dir = str(paths[settings.assistant_key])
    if settings.rag.mode != 'none':
        settings.rag.embedding_dir = str(paths[settings.rag.embedding_key])
    if settings.rag.mode == 'dense_rerank':
        settings.rag.reranker_dir = str(paths[settings.rag.reranker_key])


def inference_extras(settings) -> str:
    primary = 'vllm' if settings.backend == 'vllm' else 'llm'
    return primary + (',rag' if settings.rag.mode != 'none' else '')
