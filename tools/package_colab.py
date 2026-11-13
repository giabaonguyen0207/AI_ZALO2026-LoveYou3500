"""Build a source-only GPU kit; exclude local archives, data, weights and history."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

SOURCE_FOLDERS = {'src', 'configs', 'notebooks', 'docs', 'tools', 'tests', 'docker'}
SOURCE_FILES = {'pyproject.toml', 'requirements.txt', 'requirements-dev.lock.txt',
                'requirements-colab.txt', 'README.md', 'AGENTS.md', 'predict.sh', 'huongdankaggle.md'}


def source_path_allowed(relative):
    """One inclusion policy for the old Git snapshot and current working tree."""
    parts = Path(relative).parts
    if not parts or any(p in {'__pycache__', '.cache', '.pytest_cache', '.ipynb_checkpoints', '.git'}
                        or p.endswith('.egg-info') for p in parts):
        return False
    if len(parts) == 1:
        return parts[0] in SOURCE_FILES
    return (parts[0] in SOURCE_FOLDERS
            and not (parts[0] == 'docs' and parts[1] in {'history', 'references', 'research'}))


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    folders = ["src", "configs", "notebooks", "docs", "tools", "tests", "docker"]
    names = ["pyproject.toml", "requirements.txt", "requirements-dev.lock.txt",
             "requirements-colab.txt", "README.md", "AGENTS.md", "predict.sh"]
    if (root / 'huongdankaggle.md').is_file():
        names.append('huongdankaggle.md')
    files = [root / name for name in names]
    for folder in folders:
        for path in sorted((root / folder).rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or not source_path_allowed(relative):
                continue
            files.append(path)
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source ZIP CRC failed")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Source ZIP hash mismatch: " + name)
    report = {"archive": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "file_count": len(files), "source_only": True, "files": hashes}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs/packages/datamate-colab-source.zip")
    args = parser.parse_args()
    report = package(root, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
