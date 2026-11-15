"""CPU checks for large-image hashing and safe Docker export; no real Docker."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tools import export_solution as export


def docker_archive(path):
    with tarfile.open(path, 'w') as archive:
        for name, data in {
            'manifest.json': json.dumps([{'Config': 'config.json', 'RepoTags': ['fixture:v1'], 'Layers': []}]).encode(),
            'config.json': b'{"os":"linux","architecture":"amd64"}',
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_archive_receipts_use_md5_of_uploaded_bytes_and_do_not_need_docker(tmp_path, monkeypatch):
    archive = docker_archive(tmp_path / 'image with spaces.tar')
    payload = archive.read_bytes()
    monkeypatch.setattr(export.subprocess, 'run', lambda *a, **kw: pytest.fail('Hashing must not call Docker'))
    report = export.write_receipts(archive, google_url='https://drive.google.com/file/d/fixture/view')
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    assert report['portal_fields']['checksum_md5'] == expected
    assert len(expected) == 32 and report['sha256'] == hashlib.sha256(payload).hexdigest()
    assert report['bytes'] == len(payload) and report['uploaded'] is False
    assert Path(str(archive) + '.md5').read_text() == expected + '  ' + archive.name + '\n'
    assert json.loads(Path(str(archive) + '.submission.json').read_text())['image'] is None


def test_hashing_never_requests_an_unbounded_or_oversized_read(tmp_path, monkeypatch):
    archive = tmp_path / 'large.tar'
    payload = b'abc' * (1024 * 1024)
    archive.write_bytes(payload)
    real_open = Path.open
    reads = []

    class Guard:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def fileno(self):
            return self.handle.fileno()
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            reads.append(size)
            return self.handle.read(size)

    monkeypatch.setattr(Path, 'open', lambda self, *a, **kw: Guard(real_open(self, *a, **kw)))
    result = export.hash_archive(archive, block_bytes=1024 * 1024)
    assert len(reads) >= 4 and result['md5'] == hashlib.md5(payload, usedforsecurity=False).hexdigest()


def test_changed_or_empty_archive_is_rejected(tmp_path, monkeypatch):
    archive = tmp_path / 'changing.tar'
    archive.write_bytes(b'payload')
    real_fstat = export.os.fstat
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        stat = real_fstat(fd)
        if calls == 2:
            from types import SimpleNamespace
            return SimpleNamespace(st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns + 1)
        return stat

    monkeypatch.setattr(export.os, 'fstat', changed)
    with pytest.raises(ValueError, match='changed'):
        export.hash_archive(archive)
    monkeypatch.setattr(export.os, 'fstat', real_fstat)
    archive.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        export.hash_archive(archive)


def test_source_tar_is_rejected_without_extracting_any_files(tmp_path):
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as handle:
        member = tarfile.TarInfo('../escape.py')
        member.size = 1
        handle.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='docker save|Unsafe'):
        export.write_receipts(archive)
    assert not Path(str(archive) + '.md5').exists()
    assert not (tmp_path.parent / 'escape.py').exists()


def mock_docker(monkeypatch, *, fail_save=False, changed_tag=False, operating_system='linux'):
    commands = []
    inspections = 0

    def run(command, **kwargs):
        nonlocal inspections
        commands.append(command)
        if command[1:3] == ['image', 'inspect']:
            inspections += 1
            metadata = {'Id': 'sha256:changed' if changed_tag and inspections == 2 else 'sha256:fixture',
                        'Os': operating_system, 'Architecture': 'amd64', 'RepoDigests': []}
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(metadata))
        assert command[1:3] == ['image', 'save']
        if fail_save:
            Path(command[4]).write_bytes(b'partial')
            raise subprocess.CalledProcessError(1, command)
        docker_archive(Path(command[4]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(export.subprocess, 'run', run)
    return commands


def test_export_records_real_image_identity_and_refuses_overwrite(tmp_path, monkeypatch):
    commands = mock_docker(monkeypatch)
    output = tmp_path / 'release with spaces.tar'
    result = export.export_image('fixture:v1', output, docker='docker-custom')
    assert result['image']['id'] == 'sha256:fixture'
    assert all(c[0] == 'docker-custom' for c in commands)
    assert not list(tmp_path.glob('*.partial'))
    commands.clear()
    with pytest.raises(ValueError, match='not overwritten'):
        export.export_image('fixture:v1', output)
    assert not commands


@pytest.mark.parametrize('failure', ['save', 'changed_tag', 'windows'])
def test_export_failure_keeps_no_final_archive_or_receipts(tmp_path, monkeypatch, failure):
    mock_docker(monkeypatch, fail_save=failure == 'save', changed_tag=failure == 'changed_tag',
                operating_system='windows' if failure == 'windows' else 'linux')
    output = tmp_path / 'release.tar'
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        export.export_image('fixture:v1', output)
    assert not output.exists() and not list(tmp_path.glob('*.partial'))
    assert not Path(str(output) + '.md5').exists()


def test_cli_explicit_archive_wins_over_environment_image(tmp_path, monkeypatch, capsys):
    archive = docker_archive(tmp_path / 'existing.tar')
    monkeypatch.setenv('DATAMATE_IMAGE_TAG', 'should-not-export:v1')
    monkeypatch.setattr(sys, 'argv',
                        ['export_solution.py', '--archive', str(archive), '--hash-block-mib', '1'])
    export.main()
    report = json.loads(capsys.readouterr().out)
    assert report['archive'] == archive.name and report['image'] is None


def test_invalid_url_or_block_size_is_rejected(tmp_path):
    archive = docker_archive(tmp_path / 'image.tar')
    with pytest.raises(ValueError, match='HTTPS'):
        export.write_receipts(archive, google_url='not-a-link')
    with pytest.raises(ValueError, match='block size'):
        export.hash_archive(archive, block_bytes=0)


def test_gzip_export_hashes_compressed_bytes_and_measures_complete_tar(tmp_path, monkeypatch):
    import gzip
    mock_docker(monkeypatch)
    output = tmp_path / 'release.tar.gz'
    result = export.export_image('fixture:v1', output)
    with gzip.open(output, 'rb') as handle:
        raw = handle.read()
    assert result['archive_structure']['uncompressed_bytes'] == len(raw)
    assert result['md5'] == hashlib.md5(output.read_bytes(), usedforsecurity=False).hexdigest()
    assert result['archive_structure']['architecture'] == 'amd64'
    assert not list(tmp_path.glob('*.partial'))
    with pytest.raises(ValueError, match='100|limit|size'):
        export.verify_archive(output, max_bytes=100)


@pytest.mark.parametrize('problem', ['missing_layer', 'arm64', 'multiple_images', 'truncated_gzip'])
def test_archive_validation_rejects_incomplete_or_wrong_image(tmp_path, problem):
    import gzip
    output = tmp_path / 'bad.tar'
    entries = [{'Config':'config.json', 'RepoTags':['fixture:v1'],
                'Layers':['missing.tar'] if problem == 'missing_layer' else []}]
    if problem == 'multiple_images':
        entries *= 2
    with tarfile.open(output, 'w') as archive:
        for name, value in {'manifest.json':entries,
                            'config.json':{'os':'linux','architecture':'arm64' if problem == 'arm64' else 'amd64'}}.items():
            payload = json.dumps(value).encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    if problem == 'truncated_gzip':
        compressed = tmp_path / 'bad.tar.gz'
        compressed.write_bytes(gzip.compress(output.read_bytes())[:-5])
        output = compressed
    with pytest.raises((ValueError, EOFError, tarfile.TarError)):
        export.write_receipts(output)
    assert not Path(str(output) + '.md5').exists()
