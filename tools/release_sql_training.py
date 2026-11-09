"""Seal a completed 3000-question SQL curation pass and build a Colab kit."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from datamate.domain import json_text,DataMateError
from datamate.models.artifacts import sha256
from datamate.models.registry import get_model
from datamate.training.finetune import tokenize_conversations
from datamate.training.sql_dataset import verify_sql_dataset


def write_rows(path,rows):
    path.write_text(''.join(json_text(r)+'\n' for r in rows),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pass-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=ROOT/'.cache/tokenizers/qwen35-9b')
    p.add_argument('--packages',type=Path,default=ROOT/'outputs/packages');a=p.parse_args()
    run=a.pass_dir.resolve();out=a.output.resolve();packages=a.packages.resolve()
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    generation_hashes=json.loads((run/'source_at_generation.json').read_text(encoding='utf-8'))
    if any(sha256(ROOT/name)!=digest for name,digest in generation_hashes.items()):
        raise DataMateError('Current source differs from completed generation pass; do not rewrite hashes to bypass it')
    if summary['selected']!=3000:raise DataMateError('Release requires an inventory of all 3000 TRAIN questions')
    if out.exists() and any(out.iterdir()):raise DataMateError('Release output must be new; do not overwrite a sealed dataset')
    out.mkdir(parents=True,exist_ok=True);packages.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    token_rows=[];partitions={};length_rejected=[]
    for part in ('train','validation'):
        candidates=[json.loads(s) for s in (run/(part+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        accepted=[]
        for index,row in enumerate(candidates,1):
            indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
            last=json.loads(row['messages'][indices[-1]]['content'])
            if last.get('action')=='final' and last.get('type')=='unanswerable':
                # Checking-only plans remain context for actual tool evidence.
                # They are not targets for SQL training; learn the grounded final.
                indices=[i for i in indices if not (json.loads(row['messages'][i]['content']).get('action')=='plan'
                         and not json.loads(row['messages'][i]['content']).get('query'))]
            row['trainable_assistant_indices']=indices
            encoded,skipped=tokenize_conversations([row],tok,16384)
            expected=len(indices)
            if skipped or len(encoded)!=expected:
                length_rejected.append({'id':row['question_id'],'business_id':row['business_id'],'error':'context_limit_16384','partition':part})
                continue
            for e in encoded:
                target=[v for v in e['labels'] if v!=-100]
                if not target:raise DataMateError('Empty supervised target')
                if tok.convert_tokens_to_ids('<|im_end|>') not in target[-3:]:raise DataMateError('Missing assistant end-of-turn supervision')
                token_rows.append({'question_id':row['question_id'],'partition':part,'total_tokens':len(e['input_ids']),
                                   'supervised_tokens':len(target),'prefix_tokens':len(e['input_ids'])-len(target)})
            accepted.append(row)
            if index%500==0:print('tokenized',part,index,flush=True)
        partitions[part]=accepted;write_rows(out/(part+'.jsonl'),accepted)
    executions=[json.loads(s) for s in (run/'execution.jsonl').read_text(encoding='utf-8').splitlines()]
    queue=[json.loads(s) for s in (run/'review_queue.jsonl').read_text(encoding='utf-8').splitlines()]+length_rejected
    accepted_ids={r['question_id'] for rows in partitions.values() for r in rows}
    rejected_ids={r['id'] for r in queue}
    if accepted_ids & rejected_ids or len(accepted_ids|rejected_ids)!=3000:raise DataMateError('Missing/duplicate question in release inventory')
    inventory=[{'id':r['id'],'business_id':r['business_id'],'partition':r['partition'],'task':r.get('recipe',{}).get('task'),
                'status':'accepted' if r['id'] in accepted_ids else 'review_queue'} for r in executions]
    write_rows(out/'inventory.jsonl',inventory)
    # Compact unresolved cases retain the source question, actual prediction,
    # gold for offline review, and reason; never enter the model's messages.
    compact=[]
    for r in queue:
        compact.append({k:r.get(k) for k in ('id','business_id','partition','question','as_of_date','recipe','error','submission','reference')})
    write_rows(out/'repair_queue.jsonl',compact)
    for part in ('train','validation'):
        ids=[r['id'] for r in inventory if r['partition']==part]
        (out/(part+'_ids.json')).write_text(json_text(ids,indent=2),encoding='utf-8')
    lengths=sorted(r['total_tokens'] for r in token_rows)
    token_report={'model':'qwen35-9b','sequences':len(token_rows),'max_length':16384,
                  'min':min(lengths),'median':lengths[len(lengths)//2],'p95':lengths[int(.95*(len(lengths)-1))],'max':max(lengths),
                  'max_supervised_tokens':max(r['supervised_tokens'] for r in token_rows),
                  'max_prefix_tokens':max(r['prefix_tokens'] for r in token_rows),
                  'skipped_overlength_rows':len(length_rejected),'assistant_only_loss_verified':True,
                  'eot_supervision_verified':True,'rows':token_rows}
    (out/'tokenization_report.json').write_text(json_text(token_report,indent=2),encoding='utf-8')
    counts={part:len(rows) for part,rows in partitions.items()}
    sql_rows=sum(any(m['role']=='assistant' and (json.loads(m['content']).get('query',{}).get('sql') or json.loads(m['content']).get('action')=='sql') for m in r['messages']) for rows in partitions.values() for r in rows)
    software={name:importlib.metadata.version(name) for name in ('transformers','tokenizers','jinja2')}
    curation_report={
        'generation_summary':summary,'release_counts':counts,'repair_queue_questions':len(queue),
        'examples_with_sql':sql_rows,
        'final_types':dict(Counter(json.loads(row['messages'][-1]['content'])['type']
                                   for rows in partitions.values() for row in rows)),
        'repair_queue_by_business':dict(Counter(row['business_id'] for row in queue)),
        'curation_software':software,
        'generation_source_hashes':generation_hashes,
        'prior_proposals_thread':'codex://threads/01a1a0a2-9d9a-7c11-8487-caea75948690',
        'validation_status':'development_holdout_previously_inspected_during_curation',
        'unanswerable_policy':'supervise grounded final; mask checking-only non-SQL plans',
        'limits':['No GPU training performed','No new model-generated trajectories',
                  'Execution match alone does not prove semantic correctness or model accuracy']}
    (out/'curation_report.json').write_text(json_text(curation_report,indent=2),encoding='utf-8')
    readme=f'''# Qwen3.5-9B SQL SFT v1 — 09/11/2026

Nguồn chỉ TRAIN allowlist. Đã xử lý đủ 3.000 câu; nhận {sum(counts.values())} mẫu: {counts['train']} train, {counts['validation']} validation; {len(queue)} câu ở repair_queue, không dùng cập nhật weights. Có {sql_rows} mẫu chứa SQL. Mẫu unanswerable có bằng chứng công cụ và không bị ép có SQL.

Mẫu do bộ biên tập theo câu hỏi/schema tạo rồi replay bằng Agent thật trên CPU; không phải trajectory mới do Gemini/Qwen sinh. Gold chỉ được mở sau khi đóng băng proposals/execution trong mỗi pass để lọc và review. Các sửa đổi tiếp theo có dựa trên lỗi đã chấm trên TRAIN, gồm cả validation phát triển; không tuyên bố validation này chưa từng được xem.

Đầu vào giữ câu hỏi, as_of_date, README, rulebook và catalog của runtime. Target là JSON action plan có query.sql/parameters, lookup/schema/SQL khám phá khi cần và final unanswerable có evidence. Quan sát công cụ lấy từ thực thi thật. Không học giá trị đáp án cuối, không dùng ID để quyết định đáp án. Chỉ loss trên assistant đúng và token kết thúc lượt. Không cắt prompt: max_length 16.384.

Model: Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Giữ cùng base revision cho LoRA/QLoRA. Profile mặc định QLoRA r=16, alpha=32, LR=5e-5, 1 epoch; đây là cấu hình pilot, chưa được đo trên GPU. Source ZIP/notebook riêng cùng release. Không thay các gói frozen trong archive.

Tách doanh nghiệp seed 42 như repo; train_ids.json/validation_ids.json là toàn bộ câu của partition, gồm cả câu đang ở queue, để đánh giá inference đầy đủ. validation.jsonl chỉ chứa nhãn đã nhận, dùng theo dõi loss; không trộn vào train.jsonl. Không coi replay khớp gold là PA của model hoặc bảo đảm tổng quát hóa.

Inference tương ứng dùng mode=single, schema_mode=code_only, require_plan=true, auto_execute_plan=true. So baseline và adapter trên cùng IDs, cùng prompt/code/mode/budget trước khi kết luận hiệu quả; PA self_review cũ không là đối chứng tương đương. Chưa train GPU hoặc đo PA/tốc độ sau fine-tune.

Files: train.jsonl, validation.jsonl, manifest.json, inventory.jsonl (đủ 3000), repair_queue.jsonl, train_ids.json, validation_ids.json, tokenization_report.json. Log SQL/tool đầy đủ ở execution.jsonl của pass phát hành, lưu riêng ngoài ZIP dataset để giảm dung lượng.
'''
    readme+='\nỞ mẫu unanswerable, plan chỉ dùng để công cụ kiểm tra thiếu dữ liệu được giữ làm context và mask khỏi loss; chỉ học quyết định final có evidence thật. Luôn dùng trainable_assistant_indices khi token hóa, như trainer đi kèm.\n'
    (out/'README.md').write_text(readme,encoding='utf-8')
    model=get_model('qwen35-9b')
    source_root=ROOT/'src/datamate'
    files={f.name:sha256(f) for f in sorted(out.iterdir()) if f.is_file()}
    runtime={f.relative_to(source_root).as_posix():sha256(f) for f in sorted(source_root.rglob('*.py'))}
    tokenizer_files={name:sha256(a.tokenizer/name) for name in ('tokenizer.json','chat_template.jinja')}
    manifest={'kind':'verified_sql_sft','release':'qwen35-9b-sql-sft-v1','created_date':'2026-11-09',
              'source_role':summary['source_role'],'source_fingerprint':summary['source_fingerprint'],
              'target_model':model.to_dict(),'target_pipeline_version':6,'inference_mode':'single','schema_mode':'code_only',
              'max_length':16384,'tokenizer_checks':{'qwen35-9b':tokenizer_files},'curation_software':software,
              'train_examples':counts['train'],'validation_examples':counts['validation'],'questions_in_inventory':3000,
              'repair_queue_questions':len(queue),'examples_with_sql':sql_rows,
              'train_businesses':summary['train_businesses'],'validation_businesses':summary['validation_businesses'],
              'split_strategy':'business_holdout','seed':42,'validation_previously_inspected':True,
              'actual_model_inference_calls':0,'gpu_training_performed':False,
              'generation_method':'deterministic_question_schema_curation_with_postexecution_gold_review',
              'checking_only_plans_before_unanswerable_are_masked':True,
              'release_tool_sha256':sha256(Path(__file__)),
              'execution_sha256':sha256(run/'execution.jsonl'),'generation_summary_sha256':sha256(run/'summary.json'),
              'files':files,'runtime_source_hashes':runtime}
    (out/'manifest.json').write_text(json_text(manifest,indent=2),encoding='utf-8')
    verify_sql_dataset(out)
    data_name='qwen35-9b-sql-sft-v1.zip';data_zip=packages/data_name
    with zipfile.ZipFile(data_zip,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file():z.write(f,f.name)
    release={'dataset_archive':data_name,'dataset_archive_sha256':sha256(data_zip),
             'dataset_manifest_sha256':sha256(out/'manifest.json'),'target_model':model.to_dict()}
    (ROOT/'configs/qwen35-9b-sql-release.json').write_text(json_text(release,indent=2),encoding='utf-8')
    source_name='datamate-qwen35-9b-sql-source-v1.zip';source_zip=packages/source_name
    with zipfile.ZipFile(source_zip,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','configs','tools'):
            for f in sorted((ROOT/folder).rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and not f.is_symlink():z.write(f,f.relative_to(ROOT).as_posix())
        for name in ('pyproject.toml','requirements.txt','requirements-colab.txt'):z.write(ROOT/name,name)
    from build_sql_colab import build
    notebook=ROOT/'notebooks/training/DataMate_Colab_Qwen35_9B_SQL_SFT.ipynb'
    build(notebook,sha256(source_zip),source_name,data_name)
    release.update(source_archive=source_name,source_archive_sha256=sha256(source_zip),notebook_sha256=sha256(notebook),
                   counts={'accepted':sum(counts.values()),'train':counts['train'],'validation':counts['validation'],'queue':len(queue),'with_sql':sql_rows},
                   token_statistics={k:v for k,v in token_report.items() if k!='rows'})
    (packages/'qwen35-9b-sql-sft-v1.release.json').write_text(json_text(release,indent=2),encoding='utf-8')
    for archive in (data_zip,source_zip):
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise DataMateError('ZIP CRC failed')
    print(json_text(release,indent=2),flush=True)


if __name__=='__main__':main()
