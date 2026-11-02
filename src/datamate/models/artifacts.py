"""Online preparation is explicit; inference only accepts verified local files."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from ..domain import DataMateError, json_text
from .registry import ModelSpec, get_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(directory: Path) -> int:
    total, seen = 0, set()
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise DataMateError("Thiếu trọng số safetensors")
    for path in weights:
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 100_000_000:
                raise DataMateError("Header safetensors quá lớn")
            header = json.loads(handle.read(size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise DataMateError("Tensor trùng tên giữa các shard")
            seen.add(name)
            total += math.prod(tensor["shape"])
    return total


def artifact_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file()
                  and path.name != "artifact_manifest.json" and ".cache" not in path.relative_to(directory).parts)


def seal_artifact(directory: str | Path, spec: ModelSpec, lineage: dict | None = None) -> dict:
    directory = Path(directory).resolve()
    count = parameter_count(directory)
    if count != spec.parameters:
        raise DataMateError(f"Số phần tử tensor {count:,} khác inventory {spec.parameters:,}; kiểm tra checkpoint trước khi dùng")
    files = {str(path.relative_to(directory)).replace("\\", "/"): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in artifact_files(directory)}
    manifest = {"format_version": 1, "source": spec.to_dict(), "tensor_parameters": count,
                "files": files, "lineage": lineage}
    (directory / "artifact_manifest.json").write_text(json_text(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_artifact(directory: str | Path, key: str) -> dict:
    directory, spec = Path(directory).resolve(), get_model(key)
    path = directory / "artifact_manifest.json"
    if not path.exists():
        raise DataMateError(f"Chưa có model đã chuẩn bị tại {directory}. Chạy datamate stage-model --model {key} --output ... trước.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source") != spec.to_dict() or manifest.get("tensor_parameters") != spec.parameters:
        raise DataMateError("Manifest model không khớp registry/revision/số tham số")
    actual = {str(p.relative_to(directory)).replace("\\", "/") for p in artifact_files(directory)}
    if actual != set(manifest.get("files", {})):
        raise DataMateError("Danh sách file model đã thay đổi sau khi seal")
    for relative, meta in manifest["files"].items():
        file = (directory / relative).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != meta["sha256"]:
            raise DataMateError(f"Hash model không khớp: {relative}")
    return manifest


def stage_model(key: str, output: str | Path) -> dict:
    """The only runtime module allowed to download weights; called by a prep CLI."""
    from huggingface_hub import snapshot_download
    spec = get_model(key)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        # Resume only artifacts from the same pinned source.
        marker = output / ".source.json"
        if not marker.exists() or json.loads(marker.read_text()) != spec.to_dict():
            raise DataMateError("Thư mục model không rỗng hoặc không cùng nguồn; chọn thư mục mới")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".source.json").write_text(json_text(spec.to_dict()), encoding="utf-8")
    snapshot_download(repo_id=spec.repo_id, revision=spec.revision, local_dir=str(output),
                      local_files_only=False,
                      allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "LICENSE*", "README.md"])
    return seal_artifact(output, spec)
