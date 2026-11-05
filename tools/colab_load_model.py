# This entire file is embedded in the inference notebook's model-loading cell.
from datamate.config import Settings
from datamate.application import create_agent
from pathlib import Path
import gc, os, runpy, subprocess, sys, time
import torch

load_started = time.monotonic()
if 'MODEL_SOURCE_DIR' not in globals():
    MODEL_SOURCE_DIR = Path(MODEL_DIR)  # Compatibility with the earlier setup cell.
MODEL_SOURCE_DIR = Path(MODEL_SOURCE_DIR)
print('[1/3] Kiểm tra bản model lưu lâu dài:', MODEL_SOURCE_DIR, flush=True)
prepare_started = time.monotonic()
prepare_env = {**os.environ, 'HF_HUB_OFFLINE':'0', 'TRANSFORMERS_OFFLINE':'0'}
if not (MODEL_SOURCE_DIR / 'artifact_manifest.json').exists():
    subprocess.run([sys.executable, '-m', 'datamate', 'stage-model', '--model', MODEL_KEY,
                    '--output', str(MODEL_SOURCE_DIR)], env=prepare_env, cwd=WORK, check=True)
print('Chuẩn bị nguồn:', round(time.monotonic()-prepare_started, 2), 'giây', flush=True)

helper_path = WORK / 'tools/colab_model_cache.py'
if not helper_path.is_file():
    raise RuntimeError('Gói source cũ: thay datamate-colab-source.zip mới rồi chạy lại ô chuẩn bị.')
print('[2/3] Chép/dùng lại model trên ổ tạm Colab', flush=True)
prepare_local_model = runpy.run_path(str(helper_path))['prepare_local_model']
MODEL_DIR = prepare_local_model(MODEL_SOURCE_DIR, Path('/content/models'), MODEL_KEY)

if 'agent' in globals():
    del agent
    gc.collect()
    torch.cuda.empty_cache()
settings = Settings(model_key=MODEL_KEY, model_dir=str(MODEL_DIR),
                    quantization=QUANTIZATION, split_root=str(TRAIN_ROOT),
                    cache_dir=str(WORK/'.cache/datamate'), result_dir=str(SAVE),
                    max_context_tokens=16384, max_new_tokens=768,
                    max_steps=10, question_timeout_seconds=240)
print('[3/3] Kiểm tra SHA-256 local một lần, rồi nạp model lên GPU', flush=True)
started = time.monotonic()
agent = create_agent(settings)
print('Kiểm tra + nạp:', round(time.monotonic()-started, 2), 'giây', flush=True)
print('Loaded:', MODEL_KEY, 'tổng thời gian ô:', round(time.monotonic()-load_started, 2), 'giây', flush=True)
print('Giữ agent này để hỏi/chạy batch; không cần chạy lại ô load.', flush=True)
