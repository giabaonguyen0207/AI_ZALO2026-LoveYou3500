"""Refresh inference cells, backing up the notebook and preserving user choices."""
import json
from pathlib import Path
import re
import uuid


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'notebooks/DataMate_Colab_Inference.ipynb'
    original=path.read_bytes()
    notebook=json.loads(original)
    folder=root/'outputs/notebook-backups';folder.mkdir(parents=True,exist_ok=True)
    backup=folder/('inference-before-refresh-'+uuid.uuid4().hex[:8]+'.ipynb')
    backup.write_bytes(original)
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    setup=cells[1]
    source=''.join(setup['source'])
    source=source.replace('str(constraints)],check=True)','str(constraints)],cwd=WORK,check=True)')
    if 'DataMate import OK' not in source:
        source+='''
import importlib
source_dir = (WORK / "src").resolve()
if not (source_dir / "datamate/__init__.py").is_file():
    raise RuntimeError("Gói source thiếu src/datamate. Thay source ZIP mới rồi chạy ô đầu.")
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()
import datamate
print("DataMate import OK:", datamate.__version__, datamate.__file__)
'''
    setup['source']=source.splitlines(keepends=True)
    diagnostic=next(c for c in cells if c.get('id')=='datamate-train-diagnostic-120')
    old_diagnostic=''.join(diagnostic['source'])
    new_diagnostic=(root/'tools/diagnose_train.py').read_text(encoding='utf-8')
    for name in ('AUTO_DISCONNECT','SAMPLE_SIZE','SAMPLING_MODE','SAMPLING_SEED','KNOWN_CASE','MAX_RUN_MINUTES'):
        match=re.search(r'^'+name+r'\s*=.*$',old_diagnostic,re.M)
        if match:
            new_diagnostic=re.sub(r'^'+name+r'\s*=.*$',lambda _:match.group(),new_diagnostic,flags=re.M)
    diagnostic['source']=new_diagnostic.splitlines(keepends=True)
    loader=next(c for c in cells if c.get('id')=='s8jYg00N6yeq')
    loader['source']=(root/'tools/colab_load_model.py').read_text(encoding='utf-8').splitlines(keepends=True)
    help_cell=next(c for c in notebook['cells'] if c.get('id')=='datamate-train-diagnostic-help')
    help_cell['source']=[
        '## Chạy lại Gemma 12B và Qwen 9B — pipeline 6\n',
        'Thay source ZIP mới trên Drive, restart kernel rồi chạy từ ô đầu. Preflight phải hiện pipeline 6/backend 3 trước khi chép/nạp trọng số; ô load dùng self_review.\n',
        'Giữ mẫu 100/seed 20261103. RUN_LIMIT=None chạy đủ, 7 để thử trước; suffix mới random100-rollback-pipeline6-Nov8th-01, output tách theo model. Không resume vào lượt cũ.\n',
        'Chạy Qwen với MODEL_KEY="qwen35-9b", Gemma với "gemma4-12b" trong các kernel riêng. Output đã lưu là lịch sử; giữ nguyên AUTO_DISCONNECT và QUANTIZATION của bạn.\n',
        'Review dùng cùng model, thêm một call ở ca thuận lợi; chưa có PA/tốc độ GPU mới. Không bật lại chuẩn bị toàn schema bằng SLM. Xem docs/shared-quality-fix-2026-11-07.md.\n']
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Backup:',backup)
    print('Updated bootstrap, loader, diagnostic and help:',path)


if __name__=='__main__':
    main()
