from __future__ import annotations

import json
from pathlib import Path
import time

from ..config import configure_offline
from ..data.split import read_jsonl
from ..domain import DataMateError, json_text
from ..models.artifacts import sha256, verify_artifact, seal_artifact
from ..models.registry import get_model
from .checkpoints import exclusive_training_run


def tokenize_conversations(rows, tokenizer, max_length):
    """Supervise assistant output only, no loss on question/rulebook/tool data."""
    encoded, skipped = [], 0
    for row in rows:
        messages = row["messages"]
        selected=row.get('trainable_assistant_indices')
        if selected is not None and (not isinstance(selected,list) or not selected or
                any(type(i) is not int or not 0<=i<len(messages) or messages[i]['role']!='assistant' for i in selected) or
                len(selected)!=len(set(selected))):
            raise DataMateError('trainable_assistant_indices cần các vị trí assistant duy nhất, hợp lệ')
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or selected is not None and index not in selected:
                continue
            prefix_text = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full_text = tokenizer.apply_chat_template(messages[:index+1], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if full_text.startswith(prefix_text):
                completion_text = full_text[len(prefix_text):]
            else:
                # The pinned Gemma template adds an empty thought channel only
                # for generation prompts. Historical assistant turns omit it.
                # Keep the inference primer, supervise content + end-of-turn.
                primer = "<|channel>thought\n<channel|>"
                anchor = prefix_text[:-len(primer)] if prefix_text.endswith(primer) else None
                if anchor is None or not full_text.startswith(anchor):
                    raise DataMateError("Chat template có prefix không hỗ trợ; cần adapter tokenizer cho revision này trước training")
                completion_text = full_text[len(anchor):]
            # At generation time the prompt is already tokenized. BPE may merge
            # its trailing whitespace with the first response character when
            # encoding the concatenated string. Encode the continuation apart
            # to preserve the exact inference prefix and loss boundary.
            prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            continuation = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
            full = prefix + continuation
            if len(full) > max_length:
                skipped += 1
                continue
            if len(full) <= len(prefix):
                continue
            encoded.append({"input_ids": full, "attention_mask": [1]*len(full),
                            "labels": [-100]*len(prefix) + full[len(prefix):]})
    return encoded, skipped


@exclusive_training_run
def train(config_path, resume=None):
    started=time.monotonic()
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    configure_offline(raw.get("cache_dir", ".cache"))
    model_key, base_dir = raw["model_key"], raw["base_dir"]
    verify_artifact(base_dir, model_key)
    data_dir = Path(raw["dataset_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_role"] != "competition_training" or set(manifest["train_businesses"]) & set(manifest["validation_businesses"]):
        raise DataMateError("Training manifest không hợp lệ hoặc rò rỉ doanh nghiệp holdout")
    if manifest.get('kind')=='curated_seed':
        from .seed import verify_seed_dataset
        verify_seed_dataset(data_dir)
        if model_key not in manifest.get('tokenizer_checks',[]):
            raise DataMateError('Dataset chưa được kiểm tokenizer cho model này')
    if manifest.get('kind')=='verified_sql_sft':
        from .sql_dataset import verify_sql_dataset
        verify_sql_dataset(data_dir)
        if model_key not in manifest.get('tokenizer_checks',{}):
            raise DataMateError('SQL dataset chưa kiểm tokenizer cho model này')
    for name, digest in manifest["files"].items():
        path=(data_dir/name).resolve()
        if not path.is_relative_to(data_dir.resolve()) or sha256(path) != digest:
            raise DataMateError("Dữ liệu training bị thay đổi sau khi chuẩn bị")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise DataMateError("Training LoRA cần GPU CUDA; dùng Colab GPU")
    vram=torch.cuda.get_device_properties(0).total_memory/(1024**3)
    if vram<raw.get('min_vram_gib',0):
        raise DataMateError(f'GPU hiện có {vram:.1f} GiB, dưới ngưỡng {raw["min_vram_gib"]} GiB của profile này. Chọn GPU lớn hơn hoặc chuẩn bị profile dữ liệu khác; không cắt ngầm prompt.')
    set_seed(raw.get("seed", 42))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kernel_dispatch = None
    if raw.get('require_fast_kernels', False):
        if model_key != 'qwen35-9b': raise DataMateError('Fast training profile chỉ hỗ trợ Qwen3.5-9B')
        from .kernels import configure_training_kernels
        kernel_dispatch = configure_training_kernels()
        print('Training kernel dispatch:', json_text(kernel_dispatch), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    if manifest.get('kind')=='verified_sql_sft':
        for name,digest in manifest['tokenizer_checks'][model_key].items():
            if sha256(Path(base_dir)/name)!=digest:
                raise DataMateError('Tokenizer/chat template khác bộ SQL đã kiểm: '+name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Check targets and lengths on CPU before allocating base weights on GPU.
    train_rows, skipped_train = tokenize_conversations(read_jsonl(data_dir / "train.jsonl"), tokenizer, raw.get("max_length", 16384))
    validation_rows, skipped_validation = tokenize_conversations(read_jsonl(data_dir / "validation.jsonl"), tokenizer, raw.get("max_length", 16384))
    if not train_rows:
        raise DataMateError("Không có mẫu train fit context. Không âm thầm cắt rulebook/tool trace")
    if (skipped_train or skipped_validation) and raw.get('reject_overlength',manifest.get('kind')=='curated_seed'):
        raise DataMateError('Có mẫu vượt context; dừng trước load model. Chuẩn bị dataset/profile mới, không tự bỏ mẫu.')
    if raw.get('smoke_longest_only',False):
        train_rows=[max(train_rows,key=lambda row:len(row['input_ids']))]
        validation_rows=[]
    output = Path(raw["output_dir"])
    from .checkpoints import (atomic_json, build_run_identity, make_checkpoint_callback,
                              prepare_run, recover_completed_checkpoint)
    run_identity=build_run_identity(raw, sha256(Path(base_dir)/'artifact_manifest.json'),sha256(data_dir/'manifest.json'))
    run_identity['compute_dtype']=str(dtype)
    checkpoint=prepare_run(output,resume,run_identity)
    resume=str(checkpoint) if checkpoint is not None else None
    if resume: print('Resume từ checkpoint hoàn tất:',resume,flush=True)
    torch.cuda.reset_peak_memory_stats()
    kwargs = {"local_files_only": True, "trust_remote_code": False, "dtype": dtype, "attn_implementation": "sdpa"}
    quantized = raw.get("method", "lora") == "qlora"
    if raw.get("method", "lora") not in {"lora", "qlora"}:
        raise DataMateError("method phải là lora hoặc qlora")
    if quantized:
        kwargs.update(device_map={"": torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = AutoModelForImageTextToText.from_pretrained(base_dir, **kwargs)
    io_dtype_report = []
    if quantized:
        model = prepare_model_for_kbit_training(model)
        if model_key == 'qwen35-9b':
            from .memory import restore_frozen_io_dtype
            io_dtype_report = restore_frozen_io_dtype(model, dtype)
    suffixes = set(raw.get("target_suffixes", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]))
    targets = [name for name, module in model.named_modules() if "language_model" in name
               and name.rsplit(".", 1)[-1] in suffixes and hasattr(module, "in_features")]
    if not targets:
        raise DataMateError("Không tìm thấy linear layers của language_model; kiểm tra kiến trúc trước fine-tune")
    model = get_peft_model(model, LoraConfig(r=raw.get("rank", 16), lora_alpha=raw.get("alpha", 32),
        lora_dropout=raw.get("dropout", 0.05), target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False
    from .memory import AssistantCollator
    suffix_logits = model_key == 'qwen35-9b'
    args = TrainingArguments(output_dir=str(output), num_train_epochs=raw.get("epochs", 2),
        per_device_train_batch_size=raw.get("batch_size", 1), per_device_eval_batch_size=1,
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 8),
        learning_rate=raw.get("learning_rate", 0.0001), warmup_steps=0.03,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=raw.get('logging_steps',1), save_steps=raw.get("save_steps", 5), save_total_limit=raw.get('save_total_limit',3),
        save_strategy='steps', save_only_model=False,
        eval_strategy="steps" if validation_rows else "no", eval_steps=raw.get("eval_steps", 50),
        max_steps=raw.get("max_steps", -1), report_to="none", seed=raw.get("seed", 42),
        optim=raw.get('optimizer','adamw_torch'), remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
        data_collator=AssistantCollator(tokenizer.pad_token_id, supervised_suffix_logits=suffix_logits), processing_class=tokenizer,
        callbacks=[make_checkpoint_callback(run_identity, raw.get('save_interval_seconds',600))])
    metrics = recover_completed_checkpoint(trainer,checkpoint,run_identity) if checkpoint is not None else None
    if metrics is None:
        metrics = trainer.train(resume_from_checkpoint=resume).metrics
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    report = {"base_source": get_model(model_key).to_dict(), "training_config": raw, "dataset_manifest": manifest,
              "training_manifest_sha256": sha256(data_dir / "manifest.json"), "metrics": metrics,
              "train_sequences": len(train_rows), "validation_sequences": len(validation_rows),
              "skipped_overlength_train": skipped_train, "skipped_overlength_validation": skipped_validation,
              "tokens_per_epoch": sum(len(row["input_ids"]) for row in train_rows),
              'longest_sequence_tokens':max(len(row['input_ids']) for row in train_rows),
              'max_supervised_tokens':max(sum(x!=-100 for x in row['labels']) for row in train_rows),
              'adapter_target_modules':targets,
              'memory_optimizations':{'supervised_suffix_logits':suffix_logits,'frozen_io_dtype':io_dtype_report},
              'global_step':trainer.state.global_step,'run_identity':run_identity,'kernel_dispatch':kernel_dispatch,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
              'peak_cuda_bytes_scope':'finalization_process' if metrics.get('completion_recovered_from_checkpoint') else 'training_process',
              'cuda_device':torch.cuda.get_device_name(0),'cuda_total_gib':vram,
              'elapsed_training_process_seconds':round(time.monotonic()-started,3),
              'smoke_longest_only':bool(raw.get('smoke_longest_only',False))}
    report['adapter_files']={name:sha256(output/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    atomic_json(output/'training_report.json',report)
    return report


def merge(model_key, base, adapter, output):
    configure_offline()
    verify_artifact(base, model_key)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise DataMateError("Thư mục merge output phải rỗng")
    report_path = Path(adapter) / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["base_source"] != get_model(model_key).to_dict():
        raise DataMateError("Adapter không cùng checkpoint base")
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForImageTextToText.from_pretrained(base, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter, local_files_only=True).save_pretrained(output)
    # Preserve exact license and full training lineage alongside the export.
    import shutil
    for path in Path(base).glob("LICENSE*"):
        shutil.copy2(path, output / path.name)
    shutil.copy2(report_path, output / report_path.name)
    manifest = seal_artifact(output, get_model(model_key), {"merged_lora": True, "training_report_sha256": sha256(output / report_path.name)})
    return {"output": str(output.resolve()), "tensor_parameters": manifest["tensor_parameters"], "source": manifest["source"]}
