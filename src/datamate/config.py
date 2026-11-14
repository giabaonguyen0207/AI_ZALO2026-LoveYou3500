from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import json

from .domain import DataMateError


@dataclass
class VLLMSettings:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80
    max_num_seqs: int = 8
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    cache_prompt_prefix: bool = False  # Preserve baseline prompt; reordering is opt-in.
    group_by_business: bool = True
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization: str = 'none'
    load_format: str = 'auto'
    enforce_eager: bool = False
    cpu_offload_gb: float = 0.0
    init_timeout_seconds: float = 600.0
    abort_timeout_seconds: float = 10.0
    shutdown_timeout_seconds: float = 20.0
    extra_engine_args: dict = field(default_factory=dict)
    extra_sampling_params: dict = field(default_factory=dict)

    def validate(self):
        for name in ('tensor_parallel_size', 'pipeline_parallel_size', 'max_num_seqs'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise DataMateError(f'vllm.{name} cần số nguyên dương')
        if self.max_num_batched_tokens is not None and (type(self.max_num_batched_tokens) is not int or self.max_num_batched_tokens < 1):
            raise DataMateError('vllm.max_num_batched_tokens cần số nguyên dương hoặc null')
        for name in ('enable_prefix_caching', 'enable_chunked_prefill', 'cache_prompt_prefix', 'group_by_business', 'enforce_eager'):
            if type(getattr(self, name)) is not bool:
                raise DataMateError(f'vllm.{name} cần boolean')
        for name in ('gpu_memory_utilization', 'cpu_offload_gb', 'init_timeout_seconds', 'abort_timeout_seconds', 'shutdown_timeout_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise DataMateError(f'vllm.{name} cần số hữu hạn không âm')
        if not 0 < self.gpu_memory_utilization <= 1 or min(self.init_timeout_seconds, self.abort_timeout_seconds, self.shutdown_timeout_seconds) <= 0:
            raise DataMateError('vllm memory fraction trong (0,1]; timeout phải dương')
        if self.dtype not in {'auto', 'half', 'float16', 'bfloat16', 'float32', 'float'}:
            raise DataMateError('vllm.dtype không hợp lệ')
        for name in ('kv_cache_dtype', 'quantization', 'load_format'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise DataMateError(f'vllm.{name} cần chuỗi không rỗng')
        if not isinstance(self.extra_engine_args, dict):
            raise DataMateError('vllm.extra_engine_args cần JSON object')
        if not isinstance(self.extra_sampling_params, dict):
            raise DataMateError('vllm.extra_sampling_params cần JSON object')
        managed_sampling = {'n', 'max_tokens', 'seed', 'stop_token_ids', 'ignore_eos', 'skip_special_tokens', 'output_kind'}
        if managed_sampling & self.extra_sampling_params.keys():
            raise DataMateError('extra_sampling_params không được ghi đè contract/token budget/seed/EOS: '+str(sorted(managed_sampling & self.extra_sampling_params.keys())))
        reserved = {'model', 'tokenizer', 'trust_remote_code', 'revision', 'tokenizer_revision', 'max_model_len', 'seed',
                    'generation_config', 'override_generation_config', 'skip_tokenizer_init',
                    'speculative_config', 'enable_lora', 'lora_modules'} | {f.name for f in fields(self) if f.name != 'extra_engine_args'}
        if reserved & self.extra_engine_args.keys():
            raise DataMateError('extra_engine_args không được ghi đè trường được quản lý: '+str(sorted(reserved & self.extra_engine_args.keys())))
        try:
            json.dumps([self.extra_engine_args, self.extra_sampling_params], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DataMateError('vllm extra options phải JSON-serializable, không có NaN/Infinity') from exc
        return self


@dataclass
class RAGSettings:
    mode: str = 'none'
    embedding_key: str = 'vietnamese-embedding'
    embedding_dir: str = 'models/vietnamese-embedding'
    reranker_key: str = 'vietnamese-reranker'
    reranker_dir: str = 'models/vietnamese-reranker'
    device: str = 'cpu'
    candidate_k: int = 12
    context_k: int = 4
    context_tokens: int = 1200
    timeout_seconds: float = 45.0

    def validate(self):
        if self.mode not in {'none', 'dense', 'dense_rerank'}:
            raise DataMateError('rag.mode khong hop le')
        if self.mode != 'none' and self.embedding_key != 'vietnamese-embedding':
            raise DataMateError('rag.embedding_key phai la checkpoint embedding da kiem chung')
        if self.mode == 'dense_rerank' and self.reranker_key != 'vietnamese-reranker':
            raise DataMateError('rag.reranker_key phai la checkpoint reranker da kiem chung')
        if self.device not in {'cpu', 'cuda', 'cuda:0', 'cuda:1'}:
            raise DataMateError('rag.device khong hop le')
        if any(type(v) is not int or v < 1 for v in (self.candidate_k, self.context_k, self.context_tokens)) or self.context_k > self.candidate_k:
            raise DataMateError('rag top-k/token budget khong hop le')
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise DataMateError('rag.timeout_seconds phai duong')
        return self


@dataclass
class Settings:
    split_root: str = "train/train"
    cache_dir: str = ".cache/datamate"
    result_dir: str = "outputs/runs/default"
    model_key: str = "qwen35-9b"
    model_dir: str = "models/qwen35-9b"
    backend: str = "transformers"
    mode: str = "single"
    assistant_key: str | None = None
    assistant_dir: str | None = None
    quantization: str = "none"
    max_steps: int = 10
    max_repeated_actions: int = 2
    max_new_tokens: int = 1024
    max_context_tokens: int = 16384
    question_timeout_seconds: float = 90.0
    sql_timeout_seconds: float = 12.0
    max_result_rows: int = 150
    wall_time_seconds: float = 21000.0  # 5h50, leaves a margin below six hours.
    # Full, model-written business descriptions are an opt-in research mode.
    # Planning is independent: turning off schema preparation must not bypass it.
    adaptive_preparation: bool = False  # Backward-compatible opt-in to full preparation.
    schema_mode: str = "code_first"  # code_first (bounded repair), code_only, llm
    require_plan: bool = True
    auto_execute_plan: bool = True
    schema_repair_timeout_seconds: float = 20.0
    schema_repair_max_new_tokens: int = 512
    preparation_timeout_seconds: float = 180.0
    preparation_max_attempts: int = 2
    preparation_max_new_tokens: int = 2048
    seed: int = 42
    question_workers: int | None = None  # auto: vLLM min(4,max_num_seqs), others 1
    vllm: VLLMSettings = field(default_factory=VLLMSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)

    @property
    def effective_workers(self):
        return self.question_workers if self.question_workers is not None else min(4, self.vllm.max_num_seqs) if self.backend == 'vllm' else 1

    def validate(self) -> Settings:
        for f in fields(Settings):
            default, value = getattr(Settings(), f.name), getattr(self, f.name)
            if type(default) is int and (type(value) is not int or value < (0 if f.name == 'seed' else 1)):
                raise DataMateError(f'{f.name} cần số nguyên '+('không âm' if f.name == 'seed' else 'dương'))
            if type(default) is float and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise DataMateError(f'{f.name} cần số hữu hạn dương')
        if isinstance(self.vllm, dict):
            try:
                self.vllm = VLLMSettings(**self.vllm)
            except TypeError as exc:
                raise DataMateError(f'vllm config: {exc}') from exc
        if not isinstance(self.vllm, VLLMSettings):
            raise DataMateError('vllm cần object cấu hình')
        self.vllm.validate()
        if isinstance(self.rag, dict):
            try:
                self.rag = RAGSettings(**self.rag)
            except TypeError as exc:
                raise DataMateError(f'rag config: {exc}') from exc
        if not isinstance(self.rag, RAGSettings):
            raise DataMateError('rag can object cau hinh')
        self.rag.validate()
        if self.rag.mode != 'none' and self.backend == 'demo':
            raise DataMateError('rag can backend co tokenizer')
        if self.backend not in {"transformers", "demo", "vllm"}:
            raise DataMateError("backend phải là transformers, vllm hoặc demo")
        if self.question_workers is not None and (type(self.question_workers) is not int or self.question_workers < 1):
            raise DataMateError('question_workers/concurrency cần số nguyên dương hoặc null (auto)')
        if self.backend == 'transformers' and self.effective_workers != 1:
            raise DataMateError('Transformers baseline chỉ hỗ trợ question_workers=1; chọn vllm để batching')
        if self.backend == 'vllm' and self.mode == 'speculative':
            raise DataMateError('vLLM adapter chưa hỗ trợ speculative assistant; chọn single/self_review')
        if self.backend == 'vllm' and self.quantization != 'none':
            raise DataMateError('Dùng vllm.quantization/--vllm-quantization; 4bit/8bit là cấu hình Transformers')
        if self.mode not in {"single", "self_review", "speculative"}:
            raise DataMateError("mode phải là single, self_review hoặc speculative")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise DataMateError("quantization phải là none, 4bit hoặc 8bit")
        if self.mode == "speculative" and (not self.assistant_key or not self.assistant_dir):
            raise DataMateError("Chế độ speculative cần assistant_key và assistant_dir")
        if self.mode != "speculative" and (self.assistant_key or self.assistant_dir):
            raise DataMateError("Chỉ khai báo assistant cho chế độ speculative")
        if self.max_steps < 1 or self.max_result_rows < 1 or self.max_new_tokens < 1:
            raise DataMateError("Ngân sách bước/token/dòng phải dương")
        if type(self.max_repeated_actions) is not int or self.max_repeated_actions < 2:
            raise DataMateError("max_repeated_actions tối thiểu 2")
        if type(self.max_new_tokens) is not int or type(self.max_context_tokens) is not int or self.max_new_tokens >= self.max_context_tokens:
            raise DataMateError('max_new_tokens và max_context_tokens phải là số nguyên; output budget < context')
        if self.max_context_tokens < 2048:
            raise DataMateError("max_context_tokens tối thiểu 2048")
        if self.question_timeout_seconds <= 0 or self.sql_timeout_seconds <= 0:
            raise DataMateError("Timeout phải dương")
        if type(self.adaptive_preparation) is not bool or self.preparation_timeout_seconds <= 0:
            raise DataMateError("adaptive_preparation phải là boolean; thời gian chuẩn bị phải dương")
        if self.schema_mode not in {'code_first', 'code_only', 'llm'}:
            raise DataMateError('schema_mode phải là code_first, code_only hoặc llm')
        if type(self.require_plan) is not bool or type(self.auto_execute_plan) is not bool:
            raise DataMateError('require_plan và auto_execute_plan phải là boolean')
        if self.adaptive_preparation and self.schema_mode == 'code_only':
            raise DataMateError('code_only không dùng adaptive_preparation')
        if self.schema_repair_timeout_seconds <= 0:
            raise DataMateError('schema_repair_timeout_seconds phải dương')
        if type(self.schema_repair_max_new_tokens) is not int or not 128 <= self.schema_repair_max_new_tokens < self.max_context_tokens:
            raise DataMateError('schema_repair_max_new_tokens trong [128,max_context_tokens)')
        if type(self.preparation_max_attempts) is not int or not 1 <= self.preparation_max_attempts <= 3:
            raise DataMateError("preparation_max_attempts trong 1..3")
        if type(self.preparation_max_new_tokens) is not int or not 256 <= self.preparation_max_new_tokens < self.max_context_tokens:
            raise DataMateError('preparation_max_new_tokens trong [256,max_context_tokens)')
        if not 0 < self.wall_time_seconds <= 21600:
            raise DataMateError("wall_time_seconds phải trong (0, 21600]")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None, *, use_env=True, **overrides) -> Settings:
        path = path or (os.environ.get('DATAMATE_CONFIG') if use_env else None)
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        if not isinstance(raw, dict):
            raise DataMateError('Cấu hình cần JSON object')
        rag = raw.get('rag', {})
        if not isinstance(rag, dict):
            raise DataMateError('rag can JSON object')
        nested = raw.get('vllm', {})
        if not isinstance(nested, dict):
            raise DataMateError('vllm cần JSON object')
        nested = dict(nested)
        if use_env:
            for name, value in environment_settings().items():
                if name == 'vllm':
                    nested.update(value)
                elif name == 'rag':
                    rag.update(value)
                else:
                    raw[name] = value
        for key, value in overrides.items():
            if value is not None:
                if key == 'vllm':
                    nested.update(asdict(value) if isinstance(value, VLLMSettings) else value)
                elif key == 'rag':
                    rag.update(asdict(value) if isinstance(value, RAGSettings) else value)
                else:
                    raw[key] = value
        raw['vllm'] = nested
        raw['rag'] = rag
        try:
            return cls(**raw).validate()
        except TypeError as exc:
            raise DataMateError(f"Cấu hình không hợp lệ: {exc}") from exc

    def to_dict(self) -> dict:
        return asdict(self)


def parse_option(value, default, name):
    """Parse CLI/environment values without guessing paths/model names as JSON."""
    if isinstance(default, str):
        return value
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise DataMateError(f'{name} cần JSON number/boolean/object/null hợp lệ') from exc
    if type(default) is bool and type(parsed) is not bool:
        raise DataMateError(f'{name} cần true hoặc false')
    if type(default) is int and type(parsed) is not int:
        raise DataMateError(f'{name} cần số nguyên')
    if isinstance(default, dict) and not isinstance(parsed, dict):
        raise DataMateError(f'{name} cần JSON object')
    return parsed


def environment_settings():
    result, nested, rag = {}, {}, {}
    for f in fields(Settings):
        name = 'DATAMATE_' + f.name.upper()
        if name in os.environ:
            default = getattr(Settings(), f.name)
            if f.name == 'vllm':
                nested.update(parse_option(os.environ[name], {}, name))
            elif f.name == 'rag':
                rag.update(parse_option(os.environ[name], {}, name))
            else:
                # Optional string paths/keys must stay literal strings.
                result[f.name] = os.environ[name] if f.name in {'assistant_key', 'assistant_dir'} else parse_option(os.environ[name], default, name)
    for f in fields(VLLMSettings):
        name = 'DATAMATE_VLLM_' + f.name.upper()
        if name in os.environ:
            nested[f.name] = parse_option(os.environ[name], getattr(VLLMSettings(), f.name), name)
    for f in fields(RAGSettings):
        name = 'DATAMATE_RAG_' + f.name.upper()
        if name in os.environ:
            rag[f.name] = parse_option(os.environ[name], getattr(RAGSettings(), f.name), name)
    if nested:
        result['vllm'] = nested
    if rag:
        result['rag'] = rag
    return result


def configure_offline(cache_root: str | Path = ".cache") -> None:
    root = Path(cache_root).resolve()
    for name, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1", "VLLM_DO_NOT_TRACK": "1", "DO_NOT_TRACK": "1",
        "HF_HOME": str(root / "huggingface"), "TORCH_HOME": str(root / "torch"),
        "XDG_CACHE_HOME": str(root), "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[name] = value
