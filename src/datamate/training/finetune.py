from __future__ import annotations

import json
from pathlib import Path

from ..config import configure_offline
from ..data.split import read_jsonl
from ..domain import DataMateError, json_text
from ..models.artifacts import sha256, verify_artifact, seal_artifact
from ..models.registry import get_model


def tokenize_conversations(rows, tokenizer, max_length):
    """Supervise assistant output only, no loss on question/rulebook/tool data."""
    encoded, skipped = [], 0
    for row in rows:
        messages = row["messages"]
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
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


def train(config_path, resume=None):
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    configure_offline(raw.get("cache_dir", ".cache"))
    model_key, base_dir = raw["model_key"], raw["base_dir"]
    verify_artifact(base_dir, model_key)
    data_dir = Path(raw["dataset_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_role"] != "competition_training" or set(manifest["train_businesses"]) & set(manifest["validation_businesses"]):
        raise DataMateError("Training manifest không hợp lệ hoặc rò rỉ doanh nghiệp holdout")
    for name, digest in manifest["files"].items():
        if sha256(data_dir / name) != digest:
            raise DataMateError("Dữ liệu training bị thay đổi sau khi chuẩn bị")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise DataMateError("Training LoRA cần GPU CUDA; dùng Colab GPU")
    set_seed(raw.get("seed", 42))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"local_files_only": True, "trust_remote_code": False, "dtype": dtype, "attn_implementation": "sdpa"}
    quantized = raw.get("method", "lora") == "qlora"
    if raw.get("method", "lora") not in {"lora", "qlora"}:
        raise DataMateError("method phải là lora hoặc qlora")
    if quantized:
        kwargs.update(device_map={"": torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = AutoModelForImageTextToText.from_pretrained(base_dir, **kwargs)
    if quantized:
        model = prepare_model_for_kbit_training(model)
    suffixes = set(raw.get("target_suffixes", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]))
    targets = [name for name, module in model.named_modules() if "language_model" in name
               and name.rsplit(".", 1)[-1] in suffixes and hasattr(module, "in_features")]
    if not targets:
        raise DataMateError("Không tìm thấy linear layers của language_model; kiểm tra kiến trúc trước fine-tune")
    model = get_peft_model(model, LoraConfig(r=raw.get("rank", 16), lora_alpha=raw.get("alpha", 32),
        lora_dropout=raw.get("dropout", 0.05), target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False
    train_rows, skipped_train = tokenize_conversations(read_jsonl(data_dir / "train.jsonl"), tokenizer, raw.get("max_length", 16384))
    validation_rows, skipped_validation = tokenize_conversations(read_jsonl(data_dir / "validation.jsonl"), tokenizer, raw.get("max_length", 16384))
    if not train_rows:
        raise DataMateError("Không có mẫu train fit context. Không âm thầm cắt rulebook/tool trace")
    class Collator:
        def __call__(self, features):
            length = max(len(feature["input_ids"]) for feature in features)
            batch = {}
            for key, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                batch[key] = torch.tensor([feature[key] + [padding]*(length-len(feature[key])) for feature in features], dtype=torch.long)
            return batch
    output = Path(raw["output_dir"])
    if output.exists() and any(output.iterdir()) and resume is None:
        raise DataMateError("output_dir đã có; dùng resume hoặc chọn thư mục mới")
    args = TrainingArguments(output_dir=str(output), num_train_epochs=raw.get("epochs", 2),
        per_device_train_batch_size=raw.get("batch_size", 1), per_device_eval_batch_size=1,
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 8),
        learning_rate=raw.get("learning_rate", 0.0001), warmup_ratio=0.03,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, save_steps=raw.get("save_steps", 50), save_total_limit=2,
        eval_strategy="steps" if validation_rows else "no", eval_steps=raw.get("save_steps", 50),
        max_steps=raw.get("max_steps", -1), report_to="none", seed=raw.get("seed", 42),
        optim="adamw_torch", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows) if validation_rows else None,
        data_collator=Collator(), processing_class=tokenizer)
    outcome = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    report = {"base_source": get_model(model_key).to_dict(), "training_config": raw, "dataset_manifest": manifest,
              "training_manifest_sha256": sha256(data_dir / "manifest.json"), "metrics": outcome.metrics,
              "train_sequences": len(train_rows), "validation_sequences": len(validation_rows),
              "skipped_overlength_train": skipped_train, "skipped_overlength_validation": skipped_validation,
              "tokens_per_epoch": sum(len(row["input_ids"]) for row in train_rows),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    (output / "training_report.json").write_text(json_text(report, indent=2), encoding="utf-8")
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
