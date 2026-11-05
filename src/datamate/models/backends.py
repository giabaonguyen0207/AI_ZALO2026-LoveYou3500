from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path
from typing import Protocol

from ..config import Settings, configure_offline
from ..domain import DataMateError
from .artifacts import verify_artifact
from .registry import inventory


class ChatBackend(Protocol):
    def generate(self, messages: list[dict], remaining_seconds: float) -> str: ...


class DemoBackend:
    """Honest UI/packaging smoke mode: never fabricates a business answer."""
    counts_as_model = False
    def generate(self, messages, remaining_seconds):
        return json.dumps({"action": "final", "type": "abstain", "reason": "Chế độ demo chưa tải LLM. Chọn checkpoint local để inference thật."}, ensure_ascii=False)


def load_target(loader, model_dir, **kwargs):
    """Preserve unexpected errors; translate known GPU placement failures."""
    try:
        return loader.from_pretrained(model_dir, **kwargs)
    except (ValueError, RuntimeError) as exc:
        message = str(exc).lower()
        if ("some modules are dispatched on the cpu or the disk" in message
                or "cuda out of memory" in message):
            raise DataMateError(
                "Model không tải vừa GPU với cấu hình hiện tại. Trên máy 8GB, thử preset "
                "'Qwen 3.5 4B · Local 4bit'; dùng Colab/GPU lớn hơn cho 9B hoặc Gemma 12B. "
                "Dừng lượt inference khác trước khi thử lại. Chưa tự bật CPU/disk offload."
            ) from exc
        raise


class TransformersBackend:
    protocol_version = 2
    def __init__(self, settings: Settings):
        self.settings = settings
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

    def generate(self, messages: list[dict], remaining_seconds: float) -> str:
        if remaining_seconds <= 0:
            raise DataMateError("Hết thời gian sinh đáp án")
        with self.lock:
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False)
            length = inputs["input_ids"].shape[-1]
            if length + self.settings.max_new_tokens > self.settings.max_context_tokens:
                raise DataMateError("Context vượt giới hạn; tăng max_context_tokens hoặc giảm dữ liệu tool. Không tự cắt rulebook.")
            inputs = inputs.to(self.model.device)
            options = {"max_new_tokens": self.settings.max_new_tokens, "do_sample": False, "num_beams": 1,
                       "max_time": max(0.1, remaining_seconds), "use_cache": True}
            if self.assistant is not None:
                options["assistant_model"] = self.assistant
            started = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            tokens = output[0, length:]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            last = int(tokens[-1]) if len(tokens) else None
            self.last_generation = {"input_tokens": int(length), "output_tokens": len(tokens),
                "seconds": round(time.monotonic()-started, 4), "effective_do_sample": False,
                "stop_reason": "eos" if last in eos else "max_new_tokens" if len(tokens)>=self.settings.max_new_tokens else "time_or_other",
                "last_token_id": last, "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "prompt_tail": self.tokenizer.decode(inputs["input_ids"][0,-48:], skip_special_tokens=False)}
            return raw


def build_backend(settings: Settings) -> ChatBackend:
    return DemoBackend() if settings.backend == "demo" else TransformersBackend(settings)
