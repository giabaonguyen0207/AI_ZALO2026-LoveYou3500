"""Package source on E:, excluding all datasets, weights, credentials and caches."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "outputs/datamate-colab-source.zip"
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
