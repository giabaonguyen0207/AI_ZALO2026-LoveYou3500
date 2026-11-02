`"""Online tokenizer/template integration check; deliberately excludes weights."""
import json
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.environ["HF_HOME"]=str(root/".cache/huggingface")
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from datamate.models.registry import get_model
from datamate.data.split import DatasetSplit
from datamate.data.ingest import Ingestor
from datamate.agent.prompts import initial_messages
from datamate.training.finetune import tokenize_conversations
from datamate.domain import json_text

split=DatasetSplit(root/"train/train")
ingestor=Ingestor(split,root/".cache/datamate")
questions={}
for question in split.questions():
    questions.setdefault(question.business_id,question)
report=[]
for key in ("qwen35-9b","gemma4-12b"):
    spec=get_model(key)
    directory=root/".cache/tokenizers"/key
    snapshot_download(spec.repo_id,revision=spec.revision,local_dir=directory,
        allow_patterns=["config.json","tokenizer*","chat_template*","vocab.json","merges.txt","*.model","special_tokens_map.json"],max_workers=2)
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    lengths=[]
    for business,question in sorted(questions.items()):
        catalog=ingestor.catalog(business).summary()
        members={name for table in catalog["tables"] for name in table.get("view_of",[])}
        catalog["tables"]=[table for table in catalog["tables"] if table["name"] not in members]
        messages=initial_messages(question,split.rulebook,split.readme(business),catalog)
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tokenizer(prompt,add_special_tokens=False)["input_ids"]
        # Validate assistant-only loss boundaries on the actual pinned template.
        sample=messages+[{"role":"assistant","content":'{"action":"sql","sql":"SELECT 1"}'}]
        encoded,skipped=tokenize_conversations([{"messages":sample}],tokenizer,100000)
        assert len(encoded)==1 and skipped==0
        lengths.append({"business_id":business,"initial_prompt_tokens":len(tokens)})
    item={"model_key":key,"revision":spec.revision,"tokenizer_template_checked":True,
          "min_initial_prompt_tokens":min(x["initial_prompt_tokens"] for x in lengths),
          "max_initial_prompt_tokens":max(x["initial_prompt_tokens"] for x in lengths),"businesses":lengths}
    print(json_text({k:v for k,v in item.items() if k!="businesses"}),flush=True)
    report.append(item)
out=root/"outputs/tokenizer-check.json"
out.write_text(json_text(report,indent=2),encoding="utf-8")
