from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path

from ..config import Settings, configure_offline
from ..domain import DataMateError
from .artifacts import verify_artifact
from .registry import inventory
from .contracts import ChatBackend, GenerationTimeoutError
from .chat_format import encode_messages, input_ids, context_budget, generation_token_settings
from .runtime import runtime_identity, template_hash


class DemoBackend:
    """Honest UI/packaging smoke mode: never fabricates a business answer."""
    counts_as_model = False
    supports_token_budget = False
    last_generation = None

    def generate(self, messages, remaining_seconds, *, max_new_tokens=None):
        if remaining_seconds <= 0:
            raise GenerationTimeoutError('Hết thời gian sinh đáp án')
        return json.dumps({"action": "final", "type": "abstain", "reason": "Chế độ demo chưa tải LLM. Chọn checkpoint local để inference thật."}, ensure_ascii=False)

    def count_tokens(self, messages):
        raise DataMateError('DemoBackend không có tokenizer; dùng checkpoint tokenizer local để đếm token')

    def runtime_info(self):
        return {'backend': 'demo', 'counts_as_model': False}

    def close(self):
        pass


def load_target(loader, model_dir, **kwargs):
    """Preserve unexpected errors; translate known GPU placement failures."""
    try:
        return loader.from_pretrained(model_dir, **kwargs)
    except (ValueError, RuntimeError) as exc:
        message = str(exc).lower()
        if ("some modules are dispatched on the cpu or the disk" in message
                or "cuda out of memory" in message):
            raise DataMateError(
                "Model không tải vừa GPU với cấu hình hiện tại. Dùng Colab/Kaggle hoặc GPU thuê "
                "có đủ VRAM cho checkpoint và context đã chọn. "
                "Dừng lượt inference khác trước khi thử lại. Chưa tự bật CPU/disk offload."
            ) from exc
        raise


class TransformersBackend:
    protocol_version = 3
    supports_token_budget = True
    counts_as_model = True
    def __init__(self, settings: Settings):
        self.settings = settings
        self._closed = False
        self.last_generation = None
        self.lock = threading.Lock()
        configure_offline(Path(settings.cache_dir).parent)
        self.inventory = inventory(settings.model_key, settings.assistant_key)
        verify_artifact(settings.model_dir, settings.model_key)
        if settings.mode == "speculative":
            verify_artifact(settings.assistant_dir, settings.assistant_key)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        except ImportError as exc:
            raise DataMateError("Thiếu thư viện LLM; cài .[llm] (và bitsandbytes nếu dùng 4bit/8bit)") from exc
        self.torch = torch
        set_seed(settings.seed)
        cuda = torch.cuda.is_available()
        if settings.quantization != "none" and not cuda:
            raise DataMateError("4bit/8bit trong cấu hình này cần CUDA; chọn none để thử CPU")
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
        if settings.dtype != 'auto':
            dtype = getattr(torch, settings.dtype)
        kwargs = {"local_files_only": True, "trust_remote_code": False, "device_map": "auto",
                  "dtype": dtype, "attn_implementation": "sdpa"}
        if settings.quantization != "none":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=settings.quantization == "4bit", load_in_8bit=settings.quantization == "8bit",
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True, trust_remote_code=False)
        # Registered target checkpoints contain multimodal outer architectures.
        # Text-only prompts are tokenized locally, without loading media processors.
        self.model = load_target(AutoModelForImageTextToText, settings.model_dir, **kwargs).eval()
        self.generation_tokens = generation_token_settings(self.tokenizer, self.model.generation_config,
            vocab_size=self.model.get_output_embeddings().weight.shape[0])
        self.assistant = None
        if settings.mode == "speculative":
            # Official Gemma MTP assistant consumes hidden states / shared KV from
            # its matching target. It is not an independent answer verifier.
            self.assistant = AutoModelForCausalLM.from_pretrained(settings.assistant_dir,
                local_files_only=True, trust_remote_code=False, device_map="auto", dtype=dtype).eval()
            self.assistant.generation_config.num_assistant_tokens = 3
            self.assistant.generation_config.num_assistant_tokens_schedule = "constant"
            self.assistant.generation_config.is_assistant = True
            self.assistant.generation_config.do_sample = False
            self.assistant.generation_config.num_beams = 1

    def generate(self, messages: list[dict], remaining_seconds: float, *, max_new_tokens: int | None = None) -> str:
        if getattr(self, '_closed', False):
            raise DataMateError('Backend đã đóng')
        if remaining_seconds <= 0:
            raise GenerationTimeoutError("Hết thời gian sinh đáp án")
        with self.lock:
            self.last_generation = None
            token_budget = self.settings.max_new_tokens if max_new_tokens is None else max_new_tokens
            inputs = encode_messages(self.tokenizer, messages, return_tensors='pt')
            length = inputs["input_ids"].shape[-1]
            context_budget(int(length), token_budget, self.settings.max_context_tokens)
            inputs = inputs.to(self.model.device)
            options = {"max_new_tokens": token_budget, "do_sample": False, "num_beams": 1,
                       "max_time": max(0.1, remaining_seconds), "use_cache": True,
                       "eos_token_id": list(self.generation_tokens['effective_eos_token_ids']),
                       "pad_token_id": self.generation_tokens['effective_pad_token_id']}
            if self.assistant is not None:
                options["assistant_model"] = self.assistant
            started = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            tokens = output[0, length:]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            eos = options['eos_token_id']
            last = int(tokens[-1]) if len(tokens) else None
            self.last_generation = {"input_tokens": int(length), "output_tokens": len(tokens),
                "seconds": round(time.monotonic()-started, 4), "effective_do_sample": False,
                "stop_reason": "eos" if last in eos else "max_new_tokens" if len(tokens)>=token_budget else "time_or_other",
                "max_new_tokens":token_budget,
                "effective_eos_token_ids":list(eos), "effective_pad_token_id":options['pad_token_id'],
                "tokenizer_eos_token_id":self.generation_tokens['tokenizer_eos_token_id'],
                "last_token_id": last, "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "prompt_tail": self.tokenizer.decode(inputs["input_ids"][0,-48:], skip_special_tokens=False)}
            return raw

    def count_tokens(self, messages):
        return len(input_ids(encode_messages(self.tokenizer, messages)))

    def runtime_info(self):
        return {**runtime_identity(self.settings), 'protocol_version': self.protocol_version,
                'inventory': self.inventory,
                'chat_template_sha256': template_hash(self.tokenizer),
                'generation_tokens': self.generation_tokens,
                'device': str(getattr(self.model, 'device', None)), 'dtype': str(getattr(self.model, 'dtype', None)),
                'closed': self._closed}

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.assistant = None
            self.model = None
            # Release our references. The process may own other models (RAG),
            # so do not globally tear down CUDA/distributed state here.


def build_backend(settings: Settings, *, remaining_seconds=None) -> ChatBackend:
    if settings.backend == 'demo':
        return DemoBackend()
    if settings.backend == 'transformers':
        return TransformersBackend(settings)
    if settings.backend == 'vllm':
        from .vllm_backend import VLLMBackend
        return VLLMBackend(settings, remaining_seconds=remaining_seconds)
    raise DataMateError(f'Backend chưa được triển khai: {settings.backend}')
