"""Export a local Docker image, or hash its archive, for the Test Solution form.

No upload, network request, model inference, or whole-archive read into RAM.
MD5 is an organizer transfer checksum; SHA256 is retained for local auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from urllib.parse import urlsplit


def hash_archive(path, *, block_bytes=8 * 1024 * 1024):
    if not 0 < block_bytes <= 64 * 1024 * 1024:
        raise ValueError('Hash block size must be in (0,64 MiB]')
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as handle:
        before = os.fstat(handle.fileno())
        while block := handle.read(block_bytes):
            md5.update(block)
            sha256.update(block)
            size += len(block)
        after = os.fstat(handle.fileno())
    if size == 0:
        raise ValueError('Image archive is empty')
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
        raise ValueError('Archive changed while hashing; retry after export finishes')
    return {'bytes': size, 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def verify_archive(path):
    """Read Docker's small manifest without extracting layer contents."""
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            if member.name.removeprefix('./') != 'manifest.json':
                continue
            if not member.isfile() or not 0 < member.size <= 1024 * 1024:
                raise ValueError('Invalid Docker archive manifest')
            with archive.extractfile(member) as handle:
                manifest = json.load(handle)
            if (not isinstance(manifest, list) or not manifest
                    or any(not isinstance(item, dict) or not isinstance(item.get('Config'), str)
                           or not isinstance(item.get('Layers'), list)
                           or any(not isinstance(layer, str) for layer in item['Layers']) for item in manifest)):
                raise ValueError('Invalid Docker archive manifest')
            return {'format': 'docker-save', 'images': len(manifest)}
    raise ValueError('Missing Docker manifest.json; use docker save, not a source ZIP or docker export')


def _atomic_text(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix=path.name + '.', suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_receipts(archive, *, google_url=None, image=None, block_bytes=8 * 1024 * 1024):
    path = Path(archive).resolve()
    if not path.is_file():
        raise ValueError('Image archive not found: ' + str(path))
    if google_url:
        url = urlsplit(google_url)
        if url.scheme != 'https' or not url.netloc:
            raise ValueError('Google URL must be a complete HTTPS link')
    structure = verify_archive(path)
    hashes = hash_archive(path, block_bytes=block_bytes)
    report = {'archive': path.name, **hashes, 'archive_structure': structure,
              'google_url': google_url, 'image': image,
              'portal_fields': {'google_url': google_url, 'checksum_md5': hashes['md5']},
              'uploaded': False, 'gpu_inference_verified': False,
              'note': 'Archive manifest/hash checks do not verify GPU inference or Google link accessibility.'}
    _atomic_text(str(path) + '.md5', hashes['md5'] + '  ' + path.name + '\n')
    _atomic_text(str(path) + '.sha256', hashes['sha256'] + '  ' + path.name + '\n')
    _atomic_text(str(path) + '.submission.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def _inspect_image(image, docker):
    result = subprocess.run([docker, 'image', 'inspect', '--format', '{{json .}}', image],
                            check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    info = json.loads(result.stdout)
    if info.get('Os') != 'linux':
        raise ValueError('Submission must use a Linux image')
    return {'id': info['Id'], 'os': info['Os'], 'architecture': info.get('Architecture'),
            'repo_digests': info.get('RepoDigests', []), 'requested_image': image}


def export_image(image, output, *, docker='docker', google_url=None, block_bytes=8 * 1024 * 1024):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new archive filename; existing files are not overwritten')
    metadata = _inspect_image(image, docker)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + '.',
                                         suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
        subprocess.run([docker, 'image', 'save', '--output', str(temporary), image],
                       check=True, stdin=subprocess.DEVNULL)
        if _inspect_image(image, docker)['id'] != metadata['id']:
            raise ValueError('Image tag changed during export; retry with a fixed image tag')
        verify_archive(temporary)
        if output.exists():
            raise ValueError('Output appeared during export; choose a new archive filename')
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return write_receipts(output, google_url=google_url, image=metadata, block_bytes=block_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--image', help='Local image tag/ID; export through Docker before hashing')
    mode.add_argument('--archive', type=Path, help='Existing docker-save archive; no Docker needed')
    parser.add_argument('--output', type=Path, help='New archive path for --image')
    parser.add_argument('--docker', default=os.environ.get('DATAMATE_DOCKER', 'docker'))
    parser.add_argument('--google-url', default=os.environ.get('DATAMATE_GOOGLE_URL'))
    parser.add_argument('--hash-block-mib', type=int, default=os.environ.get('DATAMATE_HASH_BLOCK_MIB', '8'))
    args = parser.parse_args()
    image, archive = args.image, args.archive
    if not image and archive is None:
        image = os.environ.get('DATAMATE_IMAGE_TAG')
        archive = os.environ.get('DATAMATE_IMAGE_ARCHIVE')
    if bool(image) == bool(archive):
        parser.error('Choose --image or --archive, or one of DATAMATE_IMAGE_TAG / DATAMATE_IMAGE_ARCHIVE')
    if not 1 <= args.hash_block_mib <= 64:
        parser.error('--hash-block-mib must be in [1,64]')
    output = args.output or os.environ.get('DATAMATE_IMAGE_TAR')
    try:
        options = {'google_url': args.google_url, 'block_bytes': args.hash_block_mib * 1024 * 1024}
        if image:
            if not output:
                parser.error('Provide --output or DATAMATE_IMAGE_TAR for image export')
            report = export_image(image, output, docker=args.docker, **options)
        else:
            if args.output:
                parser.error('--output is only used when exporting --image')
            report = write_receipts(archive, **options)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'Solution export: {exc}\n')


if __name__ == '__main__':
    main()
