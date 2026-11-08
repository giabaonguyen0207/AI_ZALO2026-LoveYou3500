"""Package source on E:, excluding all datasets, weights, credentials and caches."""
from pathlib import Path
import argparse
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'outputs/datamate-colab-source.zip')
    output=parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    folders = ["src", "configs", "notebooks", "docs", "tools", "tests", "docker"]
    files = ["pyproject.toml", "requirements.txt", "requirements.lock.txt", "requirements-colab.txt", "README.md", "AGENTS.md", "predict.sh"]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in sorted((root / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and not any(part.endswith(".egg-info") for part in path.parts):
                    archive.write(path, path.relative_to(root))
        for name in files:
            archive.write(root / name, name)
    print(output)


if __name__=='__main__': main()
