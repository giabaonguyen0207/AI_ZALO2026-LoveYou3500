"""Prepare model + wheels online, or verify/reinstall an existing bundle offline."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
from datamate.cli import add_config, load_settings
from datamate.domain import DataMateError
from tools.runtime_artifacts import install_libraries, prepare_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config(parser)
    parser.add_argument('--artifact-root', default=os.environ.get('DATAMATE_ARTIFACT_ROOT'))
    parser.add_argument('--python', default=os.environ.get('DATAMATE_RUNTIME_PYTHON', sys.executable))
    parser.add_argument('--requirements', type=Path, help='Exact GPU lock; omit for candidate project resolution')
    parser.add_argument('--install', action='store_true', help='Install/reconcile target Python libraries from local wheels')
    parser.add_argument('--offline', action='store_true', help='Verify existing bundle only; never download or repair it')
    parser.add_argument('--index-url')
    parser.add_argument('--extra-index-url', action='append', default=[])
    args = parser.parse_args()
    try:
        if not args.artifact_root:
            raise DataMateError('Truyền --artifact-root hoặc DATAMATE_ARTIFACT_ROOT')
        root = Path(args.artifact_root).resolve()
        if args.offline:
            bundle = verify_bundle(root / 'bundle.json')
            if args.install:
                install_libraries(root / bundle['wheelhouse'], args.python)
        else:
            bundle = prepare_bundle(root, PROJECT, load_settings(args), args.python,
                requirements=args.requirements, install=args.install,
                index_url=args.index_url, extra_index_urls=args.extra_index_url)
        print(json.dumps({'bundle': str(root / 'bundle.json'), 'config': str(root / bundle['config']),
                          'wheelhouse': str(root / bundle['wheelhouse']), 'inventory': bundle['inventory'],
                          'offline_verified': args.offline}, ensure_ascii=False, indent=2))
    except subprocess.CalledProcessError as exc:
        parser.exit(2, f'Runtime preparation subprocess failed (exit {exc.returncode}); xem output phía trên.\n')
    except (DataMateError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Runtime preparation: {exc}\n')


if __name__ == '__main__':
    main()
