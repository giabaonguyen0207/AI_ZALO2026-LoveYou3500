"""Export readable step logs from a saved run; no LLM or dataset access."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main():
    source_root = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(source_root))
    from datamate.debug_log import export_debug_logs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--zip', action='store_true', help='Also package the readable logs in a ZIP')
    args = parser.parse_args()
    destination, count = export_debug_logs(args.run_dir, args.output_dir)
    print(f'Đã xuất {count} log: {destination / "index.md"}')
    if args.zip:
        output = destination.with_name(destination.name + '.zip')
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(destination.glob('*.md')):
                archive.write(path, path.name)
        print(f'ZIP: {output}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
