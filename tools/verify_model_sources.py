"""Fetch small public metadata only; never downloads model weight payloads."""
import json
import argparse
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.models.registry import MODELS, CUTOFF
from datamate.domain import json_text


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "DataMate-provenance/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def main():
    root = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'outputs/runs/model-verification' / now.strftime('%Y%m%d-%H%M%S') / 'model-evidence.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evidence output already exists; choose a new path')
    output = {"checked_on": now.date().isoformat(), "scope": "Public repository metadata and file LFS hashes, not local weight validation", "models": []}
    for spec in MODELS.values():
        url = f"https://huggingface.co/api/models/{spec.repo_id}/revision/{spec.revision}?blobs=true"
        info = fetch(url)
        history_url = f"https://huggingface.co/api/models/{spec.repo_id}/commits/{spec.revision}"
        history = fetch(history_url)
        assert info["sha"] == spec.revision
        assert info["safetensors"]["total"] == spec.parameters
        assert info["cardData"]["license"] == spec.license
        assert history[0]["id"] == spec.revision
        assert datetime.fromisoformat(history[0]["date"].replace("Z", "+00:00")) == datetime.fromisoformat(spec.revision_date.replace("Z", "+00:00"))
        weights = [file for file in info["siblings"] if file["rfilename"].endswith(".safetensors")]
        assert weights and all(file.get("lfs", {}).get("sha256") for file in weights)
        output["models"].append({"spec": spec.to_dict(), "metadata_url": url, "history_url": history_url,
            "pinned_commit": {key: history[0][key] for key in ("id", "date", "title")}, "weight_files": weights})
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(output, indent=2), encoding="utf-8")
    print(json_text({"saved": str(path), "models": len(output["models"])}))


if __name__ == "__main__":
    main()
