"""Build a participant-only Team200 RAG kit and its dedicated Colab notebook."""
from __future__ import annotations

import hashlib
import json
import runpy
import zipfile
from pathlib import Path

from create_team200_colab import BATCH, INSTALL, LOAD, SETUP, cell


DESCRIPTION = """# DataMate Team200 RAG - participant-only

This notebook follows `participant/README.md`: it verifies the frozen participant
package, reads the 200 questions in file order, writes every answer immediately,
and produces `submission.jsonl` plus `run_metadata.json`.

Upload only `datamate-team200-rag-colab-kit.zip` to `MyDrive/DataMate/`. The kit
contains the participant data and the matching source. It intentionally contains
no coordinator answers. Set `RAG_PRESET` to `rag-none`, `rag-dense`, or
`rag-dense-rerank`; set `RUN_LIMIT=None` for the complete 200-question run.
"""


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_source(root: Path, output: Path) -> None:
    helpers = ["infer_team200.py", "diagnose_train.py", "colab_model_cache.py", "colab_runtime_check.py"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "src").rglob("*.py")):
            archive.write(path, path.relative_to(root).as_posix())
        for name in helpers:
            archive.write(root / "tools" / name, "tools/" + name)
        for path in sorted((root / "configs" / "experiments" / "thinh").glob("rag-*.json")):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("tools/__init__.py", '"""Notebook runtime helpers."""\n')
        for name in ("pyproject.toml", "requirements.txt", "README.md"):
            archive.write(root / name, name)


def make_participants(root: Path, output: Path) -> None:
    participant = root / "participant"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(participant.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(root).as_posix())


def build(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    participant = root / "participant"
    verifier = runpy.run_path(str(participant / "benchmark.py"))
    manifest, questions = verifier["verify"](participant)
    if manifest["package_role"] != "participant" or len(questions) != 200:
        raise ValueError("Can participant Team200 da khoa, dung 200 cau")
    output.mkdir(parents=True, exist_ok=True)
    source, participants = output / "datamate-team200-rag-source.zip", output / "participants.zip"
    make_source(root, source)
    make_participants(root, participants)
    expected = {path.name: digest(path) for path in (source, participants)}
    setup = SETUP.replace('KIT_ZIP = BASE / "datamate-team200-colab-kit.zip"',
                          'KIT_ZIP = BASE / "datamate-team200-rag-colab-kit.zip"')
    setup = setup.replace('Đưa datamate-team200-colab-kit.zip lên Drive',
                          'Đưa datamate-team200-rag-colab-kit.zip lên Drive')
    setup = setup.replace('REFERENCE_ZIP = INPUTS / "coordinator-private.zip"  # Chỉ mở sau vòng inference để review.',
                          'REFERENCE_ZIP = None  # Participant run has no coordinator answers.')
    setup = setup.replace('SOURCE_SHA256 = EXPECTED["datamate-team200-source.zip"]',
                          'SOURCE_SHA256 = EXPECTED["datamate-team200-rag-source.zip"]')
    setup = setup.replace('extract_checked(INPUTS / "datamate-team200-source.zip", WORK)',
                          'extract_checked(INPUTS / "datamate-team200-rag-source.zip", WORK)')
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "colab": {"name": "DataMate_Colab_Team200_RAG.ipynb"}, "accelerator": "GPU"},
        "cells": [cell("markdown", DESCRIPTION, "team200-rag-intro"),
                  cell("code", setup.replace("__EXPECTED_ARCHIVES__", json.dumps(expected, indent=2)), "team200-rag-setup"),
                  cell("code", INSTALL, "team200-rag-install"),
                  cell("code", LOAD, "team200-rag-load"),
                  cell("code", BATCH, "team200-rag-run")],
    }
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
    notebook_path = output / "DataMate_Colab_Team200_RAG.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    kit = output / "datamate-team200-rag-colab-kit.zip"
    with zipfile.ZipFile(kit, "w", zipfile.ZIP_STORED) as archive:
        archive.write(source, source.name)
        archive.write(participants, participants.name)
        archive.write(notebook_path, notebook_path.name)
    if zipfile.ZipFile(kit).testzip() is not None:
        raise ValueError("Team200 RAG kit CRC failed")
    return {"kit": str(kit), "notebook": str(notebook_path), "benchmark_id": manifest["benchmark_id"], "questions": len(questions)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build(root, root / "outputs" / "packages" / "team200-rag"), ensure_ascii=False, indent=2))
