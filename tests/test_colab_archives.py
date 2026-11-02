import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from datamate.training.dataset import source_manifest
from tools.colab_archives import prepare_train_archive, verify_train_archive


def approved(split_root, tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [{
        **source_manifest(split_root), "role": "competition_training"
    }]}), encoding="utf-8")
    return path


def make_zip(split_root, destination, prefixes=("train",), backslashes=False, skip=None):
    with zipfile.ZipFile(destination, "w") as archive:
        for prefix in prefixes:
            for path in split_root.rglob("*"):
                if path.is_file() and path.relative_to(split_root).as_posix() != skip:
                    name = (Path(prefix) / path.relative_to(split_root)).as_posix()
                    archive.writestr(name.replace("/", "\\") if backslashes else name, path.read_bytes())


@pytest.mark.parametrize("prefix", ["", "train", "train/train"])
def test_train_archive_layouts_and_old_extractions(split_root, tmp_path, prefix):
    sources = approved(split_root, tmp_path)
    archive = tmp_path / "train.zip"
    make_zip(split_root, archive, prefixes=(prefix,))
    workspace = tmp_path / "work"
    stale = workspace / "uploaded-train/old-train"
    stale.mkdir(parents=True)
    (stale / "questions.jsonl").write_text("old")
    (stale / "answers.jsonl").write_text("old")
    (stale / "rulebook.md").write_text("old")
    first = prepare_train_archive(archive, workspace, sources)
    second = prepare_train_archive(archive, workspace, sources)
    assert first != second
    assert source_manifest(first) == source_manifest(split_root)
    assert source_manifest(second) == source_manifest(split_root)
    assert (stale / "questions.jsonl").read_text() == "old"


def test_windows_zip_paths_are_extracted_as_directories(split_root, tmp_path):
    sources = approved(split_root, tmp_path)
    archive = tmp_path / "train.zip"
    make_zip(split_root, archive, backslashes=True)
    extracted = prepare_train_archive(archive, tmp_path / "work", sources)
    assert (extracted / "data/shop_a/sales.csv").is_file()
    assert source_manifest(extracted) == source_manifest(split_root)


@pytest.mark.parametrize("skip, message", [
    ("answers.jsonl", "thiếu answers.jsonl"),
    ("data/shop_a/sales.csv", "Thiếu: data/shop_a/sales.csv"),
])
def test_incomplete_zip_names_missing_file(split_root, tmp_path, skip, message):
    sources = approved(split_root, tmp_path)
    archive = tmp_path / "train.zip"
    make_zip(split_root, archive, skip=skip)
    with pytest.raises(ValueError, match=message):
        verify_train_archive(archive, sources)


def test_multiple_splits_are_reported(split_root, tmp_path):
    sources = approved(split_root, tmp_path)
    archive = tmp_path / "train.zip"
    make_zip(split_root, archive, prefixes=("first/train", "second/train"))
    with pytest.raises(ValueError, match="ZIP chứa 2 split"):
        verify_train_archive(archive, sources)


def test_source_zip_instead_of_train_is_reported(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("README.md", "source only")
    with pytest.raises(ValueError, match="Không thấy questions.jsonl"):
        prepare_train_archive(archive, tmp_path / "work", tmp_path / "not-read.json")


def test_backslash_parent_path_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("train\\..\\outside.txt", "not extracted")
    with pytest.raises(ValueError, match="không an toàn"):
        prepare_train_archive(archive, tmp_path / "work", tmp_path / "not-read.json")
    assert not (tmp_path / "work").exists()


def test_packager_replaces_bad_zip_and_reuses_valid_zip(split_root, tmp_path):
    sources = approved(split_root, tmp_path)
    archive = tmp_path / "train.zip"
    archive.write_bytes(b"incomplete previous archive")
    script = Path(__file__).resolve().parents[1] / "tools/package_train.py"
    command = [sys.executable, "-X", "utf8", str(script),
               "--split-root", str(split_root), "--output", str(archive),
               "--sources", str(sources)]
    # Another reader may hold a CSV while the archive is created.
    with (split_root / "data/shop_a/sales.csv").open("rb"):
        created = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert created.returncode == 0, created.stdout + created.stderr
    assert "TRAIN ZIP OK" in created.stdout
    original = archive.read_bytes()
    reused = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert reused.returncode == 0, reused.stdout + reused.stderr
    assert "dùng lại" in reused.stdout
    assert archive.read_bytes() == original
