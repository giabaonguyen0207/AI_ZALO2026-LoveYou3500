"""Shared local async vLLM engine behind the synchronous ChatBackend port.

Only tokenization is locked. Requests from independent Agent workers run on one
event loop concurrently, allowing vLLM to continuously batch their token steps.
No HTTP server, implicit downloads, model-specific paths or token IDs.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError, TimeoutError as FutureTimeoutError
import hashlib
import inspect
from pathlib import Path
import threading
import time
import uuid

from ..config import Settings, configure_offline
from ..domain import DataMateError
from .artifacts import verify_artifact
from .chat_format import context_budget, encode_messages, generation_token_settings, input_ids
from .contracts import BackendUnavailableError, GenerationTimeoutError
from .registry import inventory
from .runtime import runtime_identity, template_hash, hardware_snapshot


def engine_options(settings):
    v = settings.vllm
    options = dict(model=str(Path(settings.model_dir).resolve()),
                   tokenizer=str(Path(settings.model_dir).resolve()), trust_remote_code=False,
                   max_model_len=settings.max_context_tokens, seed=settings.seed,
                   generation_config='vllm', tensor_parallel_size=v.tensor_parallel_size,
                   pipeline_parallel_size=v.pipeline_parallel_size,
                   gpu_memory_utilization=v.gpu_memory_utilization, max_num_seqs=v.max_num_seqs,
                   enable_prefix_caching=v.enable_prefix_caching, enable_chunked_prefill=v.enable_chunked_prefill,
                   dtype=settings.dtype if settings.dtype != 'auto' else v.dtype, kv_cache_dtype=v.kv_cache_dtype,
                   quantization=None if v.quantization == 'none' else v.quantization,
                   load_format=v.load_format, enforce_eager=v.enforce_eager,
                   cpu_offload_gb=v.cpu_offload_gb,
                   # ChatBackend supplies token IDs for text only. Qwen/Gemma
                   # outer configs also advertise vision; do not initialize
                   # media processors or encoder budgets for these requests.
                   language_model_only=True)
    if v.max_num_batched_tokens is not None:
        options['max_num_batched_tokens'] = v.max_num_batched_tokens
    options.update(v.extra_engine_args)
    return options


async def _await_if_needed(value):
    return await value if inspect.isawaitable(value) else value


class VLLMBackend:
    protocol_version = 3
    counts_as_model = True
    supports_token_budget = True
    supports_concurrency = True

    def __init__(self, settings: Settings, *, engine_factory=None, sampling_factory=None,
                 tokenizer=None, generation_config=None, remaining_seconds=None):
        started = time.monotonic()
        settings.validate()
        allowance = min(settings.vllm.init_timeout_seconds, settings.wall_time_seconds,
                        remaining_seconds if remaining_seconds is not None else settings.wall_time_seconds)
        self._startup_deadline = started + allowance
        self.startup_phases = {'artifact_verification_seconds': 0.0, 'library_import_seconds': 0.0,
                               'tokenizer_seconds': 0.0, 'engine_init_seconds': 0.0}
        if settings.backend != 'vllm':
            raise DataMateError('VLLMBackend cần backend=vllm')
        self.settings = settings
        self.inventory = inventory(settings.model_key, settings.assistant_key)
        configure_offline(Path(settings.cache_dir).parent)
        phase = time.monotonic()
        verify_artifact(settings.model_dir, settings.model_key)
        self.startup_phases['artifact_verification_seconds'] = time.monotonic()-phase
        self._init_remaining()
        self._local = threading.local()
        self._tokenizer_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed, self._fatal = False, None
        self._engine = None
        self._requests = {}  # accessed only by the engine event loop
        self._peak_requests = 0
        self.options = engine_options(settings)
        try:
            phase = time.monotonic()
            if tokenizer is None or generation_config is None:
                from transformers import AutoConfig, AutoTokenizer, GenerationConfig
                tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True, trust_remote_code=False)
                if (Path(settings.model_dir) / 'generation_config.json').is_file():
                    generation_config = GenerationConfig.from_pretrained(settings.model_dir, local_files_only=True)
                else:
                    config = AutoConfig.from_pretrained(settings.model_dir, local_files_only=True, trust_remote_code=False)
                    generation_config = GenerationConfig.from_model_config(config)
            self.startup_phases['tokenizer_seconds'] = time.monotonic()-phase
            self._init_remaining()
            phase = time.monotonic()
            if engine_factory is None:
                from vllm import SamplingParams
                from vllm.sampling_params import RequestOutputKind
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                engine_factory = lambda options: AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**options))
                sampling_factory = lambda **kwargs: SamplingParams(output_kind=RequestOutputKind.FINAL_ONLY, **kwargs)
            if sampling_factory is None:
                raise DataMateError('Engine factory cần sampling factory')
            self.startup_phases['library_import_seconds'] = time.monotonic()-phase
            self._init_remaining()
        except ImportError as exc:
            raise BackendUnavailableError('Thiếu stack vLLM/Transformers tương thích; cài extra vllm trên Linux GPU') from exc
        self.tokenizer = tokenizer
        self.generation_tokens = generation_token_settings(tokenizer, generation_config)
        self._sampling_factory = sampling_factory
        self._make_sampling(settings.max_new_tokens)  # Validate before loading engine weights.
        self._init_remaining()
        phase = time.monotonic()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve_loop, name='datamate-vllm', daemon=True)
        self._thread.start()

        async def boot():
            self._init_remaining()
            value = engine_factory(self.options)
            self._engine = await value if inspect.isawaitable(value) else value
            if self._closed:
                await self._shutdown_engine()
                return
            for method in ('generate', 'abort', 'shutdown'):
                if not callable(getattr(self._engine, method, None)):
                    raise BackendUnavailableError(f'vLLM API thiếu {method}; dùng AsyncLLM API được hỗ trợ')

        future = asyncio.run_coroutine_threadsafe(boot(), self._loop)
        try:
            future.result(timeout=self._init_remaining())
            self.startup_phases['engine_init_seconds'] = time.monotonic()-phase
        except Exception as exc:
            future.cancel()
            self._fatal = f'Khởi tạo vLLM thất bại: {exc}'
            try:
                self.close()
            except BackendUnavailableError as cleanup:
                self._fatal += f'; cleanup chưa hoàn tất: {cleanup}; cần process watchdog'
            raise BackendUnavailableError(self._fatal) from exc

    def _init_remaining(self):
        remaining = self._startup_deadline-time.monotonic()
        if remaining <= 0:
            raise BackendUnavailableError('Hết deadline/thời gian startup budget vLLM')
        return remaining

    def _serve_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    @property
    def last_generation(self):
        # Each Agent thread sees only the metadata of its own most recent call.
        return getattr(self._local, 'generation', None)

    def _check_alive(self):
        if self._fatal:
            raise BackendUnavailableError(self._fatal)
        if self._closed:
            raise BackendUnavailableError('vLLM backend đã đóng')

    def count_tokens(self, messages):
        self._check_alive()
        with self._tokenizer_lock:
            return len(input_ids(encode_messages(self.tokenizer, messages)))

    def _make_sampling(self, budget):
        options = {'n': 1, 'temperature': 0, 'seed': self.settings.seed, 'max_tokens': budget,
                   'stop_token_ids': list(self.generation_tokens['effective_eos_token_ids']),
                   'ignore_eos': False, 'skip_special_tokens': False}
        options.update(self.settings.vllm.extra_sampling_params)
        try:
            return self._sampling_factory(**options), options
        except (ValueError, TypeError) as exc:
            raise DataMateError(f'vLLM SamplingParams không hợp lệ/không tương thích: {exc}') from exc

    async def _abort(self, request_id):
        try:
            await asyncio.wait_for(_await_if_needed(self._engine.abort(request_id)),
                                   timeout=self.settings.vllm.abort_timeout_seconds)
        except Exception as exc:
            self._fatal = f'Không xác nhận abort request vLLM: {exc}'
            raise BackendUnavailableError(self._fatal) from exc

    async def _generate(self, ids, sampling, request_id, deadline):
        self._check_alive()
        self._requests[request_id] = asyncio.current_task()
        self._peak_requests = max(self._peak_requests, len(self._requests))
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GenerationTimeoutError('Hết thời gian trước khi gửi request vLLM')
            last = None
            try:
                async with asyncio.timeout(remaining):
                    async for result in self._engine.generate(
                        {'prompt_token_ids': ids}, sampling, request_id=request_id
                    ):
                        last = result
            except (TimeoutError, asyncio.CancelledError) as exc:
                await self._abort(request_id)
                raise GenerationTimeoutError('Request vLLM đã hết thời gian/hủy và được abort') from exc
            if last is None or not last.outputs:
                raise BackendUnavailableError('vLLM không trả completion')
            if not getattr(last, 'finished', False):
                await self._abort(request_id)
                raise BackendUnavailableError('vLLM đóng stream khi request chưa finished')
            # Request-level cache counts come from vLLM, not inferred from
            # overlapping callers or repeated text. None means unavailable.
            return last.outputs[0], getattr(last, 'num_cached_tokens', None)
        except (GenerationTimeoutError, BackendUnavailableError):
            raise
        except ValueError as exc:
            await self._abort(request_id)
            raise DataMateError(f'vLLM từ chối request: {exc}') from exc
        except Exception as exc:
            self._fatal = f'vLLM engine lỗi nghiêm trọng: {type(exc).__name__}: {exc}'
            raise BackendUnavailableError(self._fatal) from exc
        finally:
            self._requests.pop(request_id, None)

    def generate(self, messages, remaining_seconds, *, max_new_tokens=None):
        self._local.generation = None
        self._check_alive()
        started = time.monotonic()
        if remaining_seconds <= 0:
            raise GenerationTimeoutError('Hết thời gian sinh đáp án')
        deadline = started + remaining_seconds
        budget = self.settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        with self._tokenizer_lock:
            ids = input_ids(encode_messages(self.tokenizer, messages))
        context_budget(len(ids), budget, self.settings.max_context_tokens)
        eos = self.generation_tokens['effective_eos_token_ids']
        sampling, sampling_options = self._make_sampling(budget)
        request_id = uuid.uuid4().hex
        future = asyncio.run_coroutine_threadsafe(self._generate(ids, sampling, request_id, deadline), self._loop)
        metadata = {'input_tokens': len(ids), 'max_new_tokens': budget, 'output_tokens': 0,
                    'effective_do_sample': sampling_options['temperature'] != 0,
                    'sampling_params': sampling_options, 'effective_eos_token_ids': list(eos),
                    'effective_pad_token_id': self.generation_tokens['effective_pad_token_id']}
        try:
            # The coroutine handles its own timeout AND abort acknowledgement.
            # This outer watchdog detects a stuck event loop/cancellation path.
            result, cached_tokens = future.result(timeout=max(0, deadline-time.monotonic()) + 2*self.settings.vllm.abort_timeout_seconds)
            tokens = list(result.token_ids)
            with self._tokenizer_lock:
                raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            metadata.update(output_tokens=len(tokens), cached_tokens=cached_tokens,
                            stop_reason=getattr(result, 'finish_reason', None),
                            engine_stop_reason=getattr(result, 'stop_reason', None),
                            last_token_id=tokens[-1] if tokens else None,
                            output_sha256=hashlib.sha256(raw.encode()).hexdigest())
            return raw
        except GenerationTimeoutError:
            metadata['stop_reason'] = 'timeout_aborted'
            raise
        except (FutureTimeoutError, FutureCancelledError) as exc:
            future.cancel()
            self._fatal = 'vLLM event loop/request cancellation không hoàn thành trong hạn; dừng backend'
            metadata['stop_reason'] = 'engine_unavailable'
            raise BackendUnavailableError(self._fatal) from exc
        except BackendUnavailableError as exc:
            self._fatal = str(exc)
            metadata['stop_reason'] = 'engine_unavailable'
            raise
        finally:
            metadata['seconds'] = round(time.monotonic()-started, 4)
            self._local.generation = metadata

    def runtime_info(self):
        return {**runtime_identity(self.settings), 'protocol_version': self.protocol_version,
                'inventory': self.inventory, 'engine_args': self.options,
                'chat_template_sha256': template_hash(self.tokenizer), 'generation_tokens': self.generation_tokens,
                'startup_phases': self.startup_phases, 'hardware': hardware_snapshot(),
                'resolved_dtype': str(getattr(getattr(self._engine, 'model_config', None), 'dtype', None)).removeprefix('torch.'),
                'active_requests_peak': self._peak_requests, 'closed': self._closed, 'fatal_error': self._fatal}

    async def _shutdown(self):
        tasks = list(self._requests.values())
        for task in tasks:
            task.cancel()
        try:
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                       self.settings.vllm.abort_timeout_seconds)
        finally:
            await self._shutdown_engine()

    async def _shutdown_engine(self):
        if self._engine is not None:
            method = self._engine.shutdown
            options = {'timeout': self.settings.vllm.shutdown_timeout_seconds} if 'timeout' in inspect.signature(method).parameters else {}
            await _await_if_needed(method(**options))

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if not self._thread.is_alive():
                return
            error = None
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(
                    timeout=self.settings.vllm.shutdown_timeout_seconds)
            except Exception as exc:
                error = exc
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=self.settings.vllm.shutdown_timeout_seconds)
            if error is not None or self._thread.is_alive():
                raise BackendUnavailableError(f'vLLM shutdown không hoàn tất: {error}')
