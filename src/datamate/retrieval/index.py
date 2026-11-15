"""Checksummed NumPy indexes; no pickle, database service, or downloads."""
import hashlib
import json
from pathlib import Path
import uuid

from ..models.artifacts import sha256
from .encoders import validate_vectors
from ..domain import DataMateError


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load(directory, identity, chunks):
    import numpy as np
    directory = Path(directory)
    try:
        manifest = json.loads((directory / 'index_manifest.json').read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            return None
        for name in ('vectors.npy', 'chunks.jsonl'):
            if sha256(directory / name) != manifest['files'][name]:
                return None
        actual = [json.loads(line) for line in (directory / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        if actual != [c.to_dict() for c in chunks]:
            return None
        return validate_vectors(np.load(directory / 'vectors.npy', allow_pickle=False), expected_rows=len(chunks))
    except (OSError, ValueError, KeyError, TypeError, DataMateError):
        return None


def save(directory, identity, chunks, vectors):
    import numpy as np
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name('.' + directory.name[:12] + '.tmp-' + uuid.uuid4().hex[:12])
    temporary.mkdir()
    np.save(temporary / 'vectors.npy', vectors, allow_pickle=False)
    (temporary / 'chunks.jsonl').write_text(''.join(json.dumps(c.to_dict(), ensure_ascii=False, allow_nan=False) + '\n' for c in chunks), encoding='utf-8')
    manifest = {'identity': identity, 'files': {name: sha256(temporary / name) for name in ('vectors.npy', 'chunks.jsonl')}}
    (temporary / 'index_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if directory.exists():
        # Retain a bad cache for diagnosis rather than recursively deleting it.
        archived = directory.with_name(directory.name[:32] + '.invalid-' + uuid.uuid4().hex[:12])
        assert archived.parent == directory.parent
        directory.rename(archived)
    temporary.rename(directory)
