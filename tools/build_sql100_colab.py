"""Build the separate Colab Pro paired validation100 notebook."""
import ast
import json
from pathlib import Path
import textwrap

from tools.build_sql_colab import build as build_base


def build(output, source_sha, source_name, data_name):
    build_base(output,source_sha,source_name,data_name)
    original=json.loads(Path(output).read_text(encoding='utf-8'))
    logger=next(c for c in original['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('def run_logged'))
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):
        s=textwrap.dedent(s).lstrip('\n');ast.parse(s)
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
    md('''# Qwen3.5 SQL SFT — baseline và fine-tune trên cùng validation100

Colab Pro: chọn **A100 40 GB và High RAM**, kernel mới. Drive root giữ `MyDrive/DataMateSQLSFT`.
Chép hai ZIP mới vào thư mục đó. Adapter full và base đã có từ training được tái sử dụng.
100 ID cố định từ 630 ID validation theo doanh nghiệp; không đổi tập giữa hai model. Đây là tập phát triển đã từng được xem khi curate, không phải test độc lập.

Chạy lần lượt: setup → kiểm artifact → merge/seal → baseline 100 → fine-tune 100 → chấm đối chứng.
Hai model dùng BF16, single/code_only, context 16K, tối đa 1536 token. Mỗi model chạy trong process riêng.
Không cấu hình conv adapter dành cho training trong inference có KV cache.

Mất runtime: chạy lại từ đầu, giữ RUN_NAME/cấu hình và đúng GPU; câu đã có record hợp lệ sẽ được bỏ qua. Thời gian toàn lượt sẽ được ghi là chưa đầy đủ nếu runtime chết cứng. Mỗi RUN_NAME chỉ chạy trên một server.
''')
    code(f'''from pathlib import Path
DRIVE_ROOT=Path("/content/drive/MyDrive/DataMateSQLSFT")
SOURCE_ZIP=DRIVE_ROOT/{source_name!r}
DATA_ZIP=DRIVE_ROOT/{data_name!r}
EXPECTED_SOURCE_SHA256={source_sha!r}
TRAINING_RUN=DRIVE_ROOT/"runs/qwen35-9b-sql-v1.3-full-01"
BASE_STORE=DRIVE_ROOT/"models/qwen35-9b-c202236"
MERGED_STORE=DRIVE_ROOT/"models/qwen35-9b-sql-v1.3-full-01-merged"
RUN_NAME="qwen35-9b-sql-v1.3-validation100-01"
assert RUN_NAME and Path(RUN_NAME).name==RUN_NAME
''')
    cells.append(logger)
    code('''
    from google.colab import drive
    drive.mount("/content/drive")
    import hashlib,json,os,stat,subprocess,sys,zipfile,time
    import importlib.metadata as metadata
    from pathlib import PurePosixPath
    from uuid import uuid4
    def file_sha(path):
        with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()
    assert SOURCE_ZIP.is_file() and DATA_ZIP.is_file(), "Chép hai ZIP validation100 vào DRIVE_ROOT"
    assert file_sha(SOURCE_ZIP)==EXPECTED_SOURCE_SHA256,"Source ZIP sai release"
    WORK=Path("/content")/("datamate-sql100-"+EXPECTED_SOURCE_SHA256[:12])
    def extract_checked(archive,destination):
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            assert sum(i.file_size for i in z.infolist())<512*1024**2
            for item in z.infolist():
                p=PurePosixPath(item.filename)
                assert not p.is_absolute() and ".." not in p.parts and "\\\\" not in item.filename and ":" not in item.filename
                assert not stat.S_ISLNK(item.external_attr>>16)
            destination.mkdir(parents=True,exist_ok=True);z.extractall(destination)
    extract_checked(SOURCE_ZIP,WORK)
    if "datamate" in sys.modules:
        assert Path(sys.modules["datamate"].__file__).resolve().is_relative_to(WORK),"Restart kernel: source cũ trong RAM"
    torch_version=metadata.version("torch")
    pins=["torch=="+torch_version]
    try:pins.append("triton=="+metadata.version("triton"))
    except metadata.PackageNotFoundError:pass
    constraints=WORK/"colab-constraints.txt";constraints.write_text("\\n".join(pins)+"\\n",encoding="utf-8")
    run_logged([sys.executable,"-u","-m","pip","install","-c",str(constraints),
                "-r",str(WORK/"configs/sql-qwen35-fast-requirements.txt"),"-e",str(WORK)+"[train]"],
               cwd=WORK,log_path=DRIVE_ROOT/"setup-logs"/("sql100-"+uuid4().hex+".log"))
    assert metadata.version("torch")==torch_version,"Torch đã thay đổi; kiểm môi trường"
    sys.path.insert(0,str(WORK));sys.path.insert(0,str(WORK/"src"))
    release=json.loads((WORK/"configs/sql100-release.json").read_text(encoding="utf-8"))
    assert file_sha(DATA_ZIP)==release["data_sha256"],"Data ZIP sai release"
    BENCH=Path("/content")/("datamate-sql100-data-"+release["data_sha256"][:12])
    extract_checked(DATA_ZIP,BENCH)
    from tools.sql100 import verify_benchmark
    manifest=verify_benchmark(BENCH)
    assert manifest["sample_size"]==100
    print("Đã khóa cùng 100 câu:",manifest["business_counts"])
    ''')
    code('''
    import torch
    from datamate.models.artifacts import verify_artifact,stage_model
    from datamate.training.checkpoints import completed_report,ensure_training_config
    from tools.sql100 import stage_local,write_json,runtime_identity
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"Cần GPU CUDA hỗ trợ BF16"
    assert torch.cuda.get_device_properties(0).total_memory/1024**3>=38,"Chọn A100 40GB để merge và so BF16"
    ADAPTER=TRAINING_RUN/"adapter"
    for name,digest in release["adapter_files"].items():
        assert file_sha(ADAPTER/name)==digest,"Adapter full khác artifact đã kiểm: "+name
    training_report=json.loads((ADAPTER/"training_report.json").read_text(encoding="utf-8"))
    assert completed_report(ADAPTER,training_report["training_config"]) is not None
    assert training_report["global_step"]==298 and training_report["metrics"]["epoch"]==1.0
    os.environ.pop("HF_HUB_OFFLINE",None);os.environ.pop("TRANSFORMERS_OFFLINE",None)
    if not (BASE_STORE/"artifact_manifest.json").exists():stage_model("qwen35-9b",BASE_STORE)
    verify_artifact(BASE_STORE,"qwen35-9b")
    assert file_sha(BASE_STORE/"artifact_manifest.json")==release["base_manifest_sha256"],"Base khác lượt train"
    RESULT=DRIVE_ROOT/"runs"/RUN_NAME;RESULT.mkdir(parents=True,exist_ok=True)
    invocation=uuid4().hex
    write_json(RESULT/("environment-"+invocation+".json"),{
        "source_sha256":EXPECTED_SOURCE_SHA256,"data_sha256":release["data_sha256"],
        "runtime":runtime_identity(),"training_report_sha256":file_sha(ADAPTER/"training_report.json")})
    (RESULT/("pip-freeze-"+invocation+".txt")).write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True),encoding="utf-8")
    print("Full adapter đã xác minh; GPU:",torch.cuda.get_device_name(0))
    ''')
    code('''
    expected_report_sha=release["adapter_files"]["training_report.json"]
    if (MERGED_STORE/"artifact_manifest.json").exists():
        merged_manifest=verify_artifact(MERGED_STORE,"qwen35-9b")
        assert merged_manifest["lineage"]["training_report_sha256"]==expected_report_sha,"Merged model khác adapter"
        assert merged_manifest["lineage"]["merged_lora"] is True
        print("Tái sử dụng merged model đã kiểm")
    else:
        if MERGED_STORE.exists() and any(MERGED_STORE.iterdir()):
            assert MERGED_STORE.resolve().parent==(DRIVE_ROOT/"models").resolve()
            preserved=MERGED_STORE.with_name(MERGED_STORE.name+"-incomplete-"+uuid4().hex[:8])
            MERGED_STORE.rename(preserved);print("Giữ bản merge ghi dở:",preserved)
        run_logged([sys.executable,"-u","-m","datamate","merge","--model","qwen35-9b",
                    "--base",str(BASE_STORE),"--adapter",str(ADAPTER),"--output",str(MERGED_STORE)],
                   cwd=WORK,log_path=RESULT/"process-logs"/("merge-"+uuid4().hex+".log"))
        merged_manifest=verify_artifact(MERGED_STORE,"qwen35-9b")
        assert merged_manifest["lineage"]["training_report_sha256"]==expected_report_sha
    # Keep model loading off Drive. Copy/verify time is reported separately.
    copied=time.monotonic()
    CACHE_MODELS=Path("/content/datamate-sql100-models")
    BASE_LOCAL=stage_local(BASE_STORE,CACHE_MODELS/"base")
    MERGED_LOCAL=stage_local(MERGED_STORE,CACHE_MODELS/"finetuned")
    write_json(RESULT/("staging-"+invocation+".json"),{"copy_verify_seconds":time.monotonic()-copied})
    configs={}
    for variant,model_dir in [("baseline",BASE_LOCAL),("finetuned",MERGED_LOCAL)]:
        config={"backend":"transformers","model_key":"qwen35-9b","model_dir":str(model_dir),
            "mode":"single","quantization":"none","split_root":str(BENCH/"input"),
            "cache_dir":str(Path("/content/sql100-cache")/RUN_NAME/variant),"result_dir":str(RESULT/variant),
            "schema_mode":"code_only","adaptive_preparation":False,"require_plan":True,"auto_execute_plan":True,
            "max_context_tokens":16384,"max_new_tokens":1536,"max_steps":8,"max_result_rows":1000,
            "sql_timeout_seconds":15,"question_timeout_seconds":90,"wall_time_seconds":21000,"seed":42}
        configs[variant]=RESULT/(variant+"-config.json")
        ensure_training_config(configs[variant],config)
    print("Model đã merge/seal. Bây giờ chạy baseline rồi fine-tune trên cùng 100 câu.")
    ''')
    for variant in ('baseline','finetuned'):
        md('### '+('Baseline — Qwen gốc' if variant=='baseline' else 'Fine-tune — adapter full đã merge')+' · 100 câu')
        code(f'''run_logged([sys.executable,"-u",str(WORK/"tools/sql100.py"),"run",
            "--config",str(configs[{variant!r}]),"--benchmark",str(BENCH)],
           cwd=WORK,log_path=RESULT/"process-logs"/({variant!r}+"-"+uuid4().hex+".log"))
''')
    md('### Chấm sau khi cả hai model đã sinh đủ 100 đáp án')
    code('''
    run_logged([sys.executable,"-u",str(WORK/"tools/sql100.py"),"report","--benchmark",str(BENCH),
                "--baseline",str(RESULT/"baseline"),"--finetuned",str(RESULT/"finetuned"),
                "--output",str(RESULT/"review")],cwd=WORK,
               log_path=RESULT/"process-logs"/("report-"+uuid4().hex+".log"))
    comparison=json.loads((RESULT/"review/comparison.json").read_text(encoding="utf-8"))
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
    bundle=RESULT/("comparison-bundle-"+uuid4().hex[:8]+".zip")
    with zipfile.ZipFile(bundle,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RESULT.rglob("*")):
            if p.is_file() and p.suffix!=".zip" and ".tmp-" not in p.name:
                z.write(p,p.relative_to(RESULT).as_posix())
    print("Kết quả trên Drive:",RESULT)
    print("Gửi comparison.json, paired_review.json và regressions.json để phân tích; ZIP đầy đủ:",bundle)
    print("Đã xong. Có thể ngắt runtime để tiết kiệm GPU. Điểm này là validation phát triển, chưa phải test BTC.")
    ''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':original['metadata'],'cells':cells}
    for i,c in enumerate(cells):c['id']=f'sql100-{i:02d}'
    Path(output).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
