"""Update only bootstrap + diagnostic cells, preserving user choices and logs."""
import json
from pathlib import Path
import uuid


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'notebooks/DataMate_Colab_Inference.ipynb'
    original=path.read_bytes()
    notebook=json.loads(original)
    folder=root/'outputs/notebook-backups';folder.mkdir(parents=True,exist_ok=True)
    backup=folder/('inference-before-pipeline-v2-'+uuid.uuid4().hex[:8]+'.ipynb')
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
    setup['execution_count'],setup['outputs']=None,[]
    diagnostic=next(c for c in cells if c.get('id')=='datamate-train-diagnostic-120')
    diagnostic['source']=(root/'tools/diagnose_train.py').read_text(encoding='utf-8').splitlines(keepends=True)
    diagnostic['execution_count'],diagnostic['outputs']=None,[]
    help_cell=next(c for c in notebook['cells'] if c.get('id')=='datamate-train-diagnostic-help')
    help_cell['source']=[
        '## Inference lần 3 · 100 câu ngẫu nhiên từ toàn bộ train\n',
        '**Trước lần chạy mới:** thay `datamate-colab-source.zip` trên Drive bằng gói mới trong outputs local. Khởi động lại kernel/runtime để nạp code mới; giữ MODEL_KEY và QUANTIZATION như baseline. Chạy các ô chuẩn bị/load, sau đó ô batch này.\n',
        'Tên lượt mới `random100-round3-01`; seed `20261103`. Bốc đều, không hoàn lại từ toàn bộ 3.000 câu train; không cân đối type/doanh nghiệp/ngôn ngữ, không lọc câu cũ hoặc holdout. Không fine-tune; trần mềm 90 phút.\n',
        'Sau dòng REPORT, kiểm tra ZIP đã lưu trên Drive rồi ngắt GPU. ZIP có log JSON và Markdown cho từng câu để xem LLM, SQL, kết quả/lỗi từng bước. Chưa có accuracy LLM của lượt 3.\n']
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Backup:',backup)
    print('Updated bootstrap and diagnostic only:',path)


if __name__=='__main__':
    main()
