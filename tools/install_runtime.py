"""Verify a staged bundle, install entirely offline, and relocate model paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
from datamate.config import Settings
from datamate.data.split import ensure_outside_dataset
from datamate.domain import DataMateError
from datamate.models.registry import relocate_models, inference_extras
from tools.runtime_artifacts import _run, _save, install_libraries, verify_bundle


def install_bundle(bundle_path, python, project, output_config, *, check_only=False):
    bundle_path = Path(bundle_path).resolve()
    bundle = verify_bundle(bundle_path)
    root = bundle_path.parent
    wheelhouse = root / bundle['wheelhouse']
    settings = Settings.load(root / bundle['config'], use_env=False)
    output = ensure_outside_dataset(output_config, settings.split_root)
    if output in {bundle_path, root / bundle['config']}:
        raise DataMateError('Chọn config local mới; không ghi đè bundle/config đã seal')
    if not (Path(project) / 'pyproject.toml').is_file():
        raise DataMateError('Project root thiếu pyproject.toml')
    result = install_libraries(wheelhouse, python, check_only=check_only)
    if check_only:
        return {'libraries': result, 'bundle_verified': True, 'config_written': False}
    _run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(wheelhouse),
          '--no-build-isolation', '--editable', str(Path(project).resolve()) + '[' + inference_extras(settings) + ']'])
    _run([str(python), '-m', 'pip', 'check'])
    relocate_models(settings, {key: str(root / item['path']) for key, item in bundle['models'].items()})
    _save(output, settings.to_dict())
    return {'libraries': result, 'bundle_verified': True, 'config': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default=os.environ.get('DATAMATE_BUNDLE'))
    parser.add_argument('--python', default=os.environ.get('DATAMATE_RUNTIME_PYTHON', sys.executable))
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--output-config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        if not args.bundle:
            raise DataMateError('Truyền --bundle hoặc DATAMATE_BUNDLE')
        path = Path(args.bundle).resolve()
        output = args.output_config or path.parent / 'runtime-local.json'
        print(json.dumps(install_bundle(path, args.python, args.project_root, output,
                                       check_only=args.check_only), ensure_ascii=False, indent=2))
    except subprocess.CalledProcessError as exc:
        parser.exit(2, f'Offline install subprocess failed (exit {exc.returncode}); xem output phía trên.\n')
    except (DataMateError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Offline runtime install: {exc}\n')


if __name__ == '__main__':
    main()
