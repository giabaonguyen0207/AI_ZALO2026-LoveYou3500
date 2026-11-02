"""Create/verify the original TRAIN ZIP without Compress-Archive file sharing."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import zipfile

from colab_archives import verify_train_archive, verify_train_directory


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=project / "train/train")
    parser.add_argument("--output", type=Path, default=project / "outputs/train.zip")
    parser.add_argument("--sources", type=Path, default=project / "configs/training_sources.json")
    args = parser.parse_args()
    source, output = args.split_root.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        parser.error("ZIP phải nằm ngoài thư mục dữ liệu TRAIN")
    try:
        manifest = verify_train_directory(source, args.sources)
        if output.is_file():
            try:
                verify_train_archive(output, args.sources)
                print(f"TRAIN ZIP OK (dùng lại): {output}")
                print(f"{len(manifest['files'])} file; fingerprint {manifest['fingerprint']}")
                return
            except (ValueError, zipfile.BadZipFile):
                print("ZIP cũ chưa hợp lệ; sẽ thay sau khi ZIP mới vượt kiểm tra.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix="train-", suffix=".zip.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(manifest["files"]):
                    archive.write(source / name, (Path("train") / name).as_posix())
            verify_train_archive(temporary, args.sources)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"TRAIN ZIP OK: {output}")
        print(f"{len(manifest['files'])} file; fingerprint {manifest['fingerprint']}")
    except OSError as exc:
        parser.exit(1, f"Không đọc/ghi được file: {exc}\nĐóng ứng dụng đang khóa file (web: Ctrl+C ở cửa sổ chạy web), rồi thử lại.\n")
    except (ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Không tạo ZIP: {exc}\n")


if __name__ == "__main__":
    main()
