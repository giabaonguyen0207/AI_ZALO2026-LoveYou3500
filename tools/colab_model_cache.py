"""Copy sealed model files to runtime disk; backend still verifies every hash."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time


def prepare_local_model(source, cache_root, model_key, *, refresh=False, log=print):
    source, cache_root = Path(source).resolve(), Path(cache_root).resolve()
    if not model_key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in model_key):
        raise ValueError('Invalid model cache key')
    manifest_bytes = (source / 'artifact_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get('files'):
        raise ValueError('Manifest model rỗng; không tạo cache')
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    destination = cache_root / model_key / digest
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Cache local phải tách khỏi nguồn model')
    files = []
    for name, meta in manifest['files'].items():
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or '..' in path.parts
                or '\\' in name or ':' in name or name == 'artifact_manifest.json'):
            raise ValueError(f'Đường dẫn manifest không hợp lệ: {name}')
        original, target = source.joinpath(*path.parts), destination.joinpath(*path.parts)
        if not original.resolve().is_relative_to(source) or not target.resolve().is_relative_to(destination):
            raise ValueError(f'File trỏ ra ngoài thư mục: {name}')
        size = meta['bytes']
        if type(size) is not int or size < 0 or not original.is_file() or original.stat().st_size != size:
            raise ValueError(f'File nguồn thiếu/khác dung lượng manifest: {name}')
        if refresh or not target.is_file() or target.stat().st_size != size:
            files.append((name, original, target, size))
    cache_root.mkdir(parents=True, exist_ok=True)
    needed = sum(row[3] for row in files) + len(manifest_bytes) + 64 * 1024**2
    free = shutil.disk_usage(cache_root).free
    if free < needed:
        raise OSError(f'Ổ tạm không đủ chỗ: cần khoảng {needed/2**30:.2f} GiB, còn {free/2**30:.2f} GiB. Không tự chuyển sang đọc Drive.')
    started = time.monotonic()
    log(f'Cache local: {destination}', flush=True)
    log(f'Cần chép {len(files)} file, {sum(row[3] for row in files)/2**30:.2f} GiB', flush=True)
    for index, (name, original, target, size) in enumerate(files, 1):
        log(f'  [{index}/{len(files)}] {name} ({size/2**30:.2f} GiB)', flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A failed copy is never published under the real filename.
        temporary = target.with_name(target.name + '.copying')
        if temporary.is_symlink():
            raise ValueError(f'Cache tạm là symlink: {temporary}')
        with original.open('rb') as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024**2)
        if temporary.stat().st_size != size:
            raise ValueError(f'Chép chưa đủ file: {name}; chạy lại ô để thử lại')
        os.replace(temporary, target)
    if (source / 'artifact_manifest.json').read_bytes() != manifest_bytes:
        raise ValueError('Manifest nguồn thay đổi lúc chép; chạy lại sau khi upload hoàn tất')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'artifact_manifest.json').write_bytes(manifest_bytes)
    log(f'Chuẩn bị cache: {time.monotonic()-started:.2f}s. SHA-256 sẽ được backend kiểm tra trước khi load.', flush=True)
    return destination
