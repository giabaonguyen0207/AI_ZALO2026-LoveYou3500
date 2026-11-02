"""Validate and extract Colab input archives using only the standard library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

REQUIRED = ("questions.jsonl", "answers.jsonl", "rulebook.md", "calendar.csv")


def _hash(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(files):
    fingerprint = hashlib.sha256(json.dumps(
        files, ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf-8")).hexdigest()
    return {"fingerprint": fingerprint, "files": files}


def _approve(manifest, sources):
    approved = json.loads(Path(sources).read_text(encoding="utf-8"))["sources"]
    training = [entry for entry in approved if entry["role"] == "competition_training"]
    if any(entry["fingerprint"] == manifest["fingerprint"] for entry in training):
        return
    details = []
    if training:
        expected, actual = training[0]["files"], manifest["files"]
        for label, names in (
            ("Thiếu", expected.keys() - actual.keys()),
            ("Thừa", actual.keys() - expected.keys()),
            ("Khác hash", {name for name in expected.keys() & actual.keys()
                          if expected[name] != actual[name]}),
        ):
            if names:
                details.append(f"{label}: {', '.join(sorted(names)[:5])}")
    raise ValueError(
        "Dữ liệu không khớp gói TRAIN đã đăng ký. " + "; ".join(details)
        + ". Tạo lại ZIP bằng tools/package_train.py; không sửa hash trong cấu hình."
    )


def verify_train_directory(root, sources):
    root = Path(root).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    data = sorted(path for path in (root / "data").rglob("*") if path.is_file())
    if missing or not data:
        raise ValueError(f"Thư mục TRAIN thiếu: {', '.join(missing + ([] if data else ['data/']))}: {root}")
    files = {}
    for path in [*(root / name for name in REQUIRED), *data]:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"File trỏ ra ngoài TRAIN: {path}")
        with path.open("rb") as stream:
            files[path.relative_to(root).as_posix()] = _hash(stream)
    manifest = _manifest(files)
    _approve(manifest, sources)
    return manifest


def _members(archive):
    members, seen = [], set()
    for info in archive.infolist():
        # Older Windows ZIP writers can store backslashes. Normalize before
        # checking paths and extracting on Linux; never extract literal '\\'.
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or any(":" in part for part in path.parts)
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Đường dẫn ZIP không an toàn: {info.filename}")
        if path in seen:
            raise ValueError(f"ZIP chứa đường dẫn trùng: {path}")
        seen.add(path)
        if not name.endswith("/"):
            members.append((info, path))
    return members


def _train_members(archive):
    members = _members(archive)
    names = {path for _, path in members}
    candidates = sorted(path.parent for path in names if path.name == "questions.jsonl")
    roots, problems = [], []
    for root in candidates:
        missing = [name for name in REQUIRED if root / name not in names]
        if not any(path.is_relative_to(root / "data") for path in names):
            missing.append("data/")
        if missing:
            problems.append(f"{root}: thiếu {', '.join(missing)}")
        else:
            roots.append(root)
    if not roots:
        details = "; ".join(problems[:5]) or "Không thấy questions.jsonl trong ZIP."
        sample = ", ".join(str(path) for _, path in members[:8]) or "(ZIP rỗng)"
        raise ValueError(
            f"Không tìm thấy split TRAIN đầy đủ trong ZIP. {details}\n"
            f"Các file đầu: {sample}\n"
            "Kiểm tra TRAIN_ZIP đang trỏ tới train.zip, không phải gói source. "
            "Tạo lại bằng tools/package_train.py rồi thay file trên Drive."
        )
    if len(roots) != 1:
        raise ValueError(
            f"ZIP chứa {len(roots)} split đầy đủ: {', '.join(map(str, roots))}. "
            "Chỉ nén thư mục train/train bằng tools/package_train.py."
        )
    root = roots[0]
    selected = [(info, path) for info, path in members
                if path in {root / name for name in REQUIRED}
                or path.is_relative_to(root / "data")]
    return root, selected


def _verify_open_archive(archive, sources):
    root, members = _train_members(archive)
    files = {}
    for info, path in members:
        with archive.open(info) as stream:
            files[path.relative_to(root).as_posix()] = _hash(stream)
    manifest = _manifest(files)
    _approve(manifest, sources)
    return root, members, manifest


def verify_train_archive(archive_path, sources):
    with zipfile.ZipFile(archive_path) as archive:
        root, members, manifest = _verify_open_archive(archive, sources)
    return {"train_prefix": str(root), "file_count": len(members), **manifest}


def prepare_train_archive(archive_path, workspace, sources):
    """Check the ZIP itself, then extract into a fresh directory on each run."""
    archive_path, workspace = Path(archive_path), Path(workspace).resolve()
    if not archive_path.is_file():
        raise ValueError(f"Chưa có file: {archive_path}. Upload train.zip hoặc sửa TRAIN_ZIP.")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            root, members, manifest = _verify_open_archive(archive, sources)
            parent = workspace / "uploaded-train"
            parent.mkdir(parents=True, exist_ok=True)
            destination = Path(tempfile.mkdtemp(prefix="train-", dir=parent))
            for info, path in members:
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"ZIP hỏng hoặc upload chưa xong: {archive_path}. {exc}") from exc
    train_root = destination.joinpath(*root.parts)
    print(f"TRAIN OK: {len(members)} file, fingerprint {manifest['fingerprint']}")
    print("Train:", train_root)
    return train_root
