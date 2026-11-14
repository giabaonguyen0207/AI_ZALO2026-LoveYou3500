"""Local Vietnamese retrieval: source-scoped corpus, reusable index and adapters."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from ..data.split import ensure_outside_dataset
from ..domain import DataMateError
from ..models.artifacts import sha256, verify_artifact
from .contracts import RetrievalFatalError, RetrievalTimeoutError
from .encoders import VietnameseEncoders, validate_vectors
from . import index as disk_index


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    scope: str
    digest: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {**self.metadata, 'text': self.text, 'source': self.source,
                'scope': self.scope, 'source_sha256': self.digest}


def sections(text, source, scope, *, tokenizer, chunk_tokens=384, overlap=48,
             source_sha256=None, metadata=None):
    """Split original text by headings and actual embedding-token budget."""
    digest = source_sha256 or hashlib.sha256(text.encode('utf-8')).hexdigest()
    groups, current = [], []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('#') and current:
            groups.append(''.join(current).strip())
            current = []
        current.append(line)
    if current:
        groups.append(''.join(current).strip())
    result = []
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    for section_number, value in enumerate(groups):
        if not value:
            continue
        heading = value.splitlines()[0] if value.startswith('#') else ''
        if count(heading) >= chunk_tokens // 2:
            heading = ''
        position = 0
        while position < len(value):
            prefix = heading + '\n' if position and heading else ''
            low, high = 1, len(value) - position
            while low < high:
                mid = (low + high + 1) // 2
                if count(prefix + value[position:position + mid]) <= chunk_tokens:
                    low = mid
                else:
                    high = mid - 1
            piece = prefix + value[position:position + low]
            if count(piece) > chunk_tokens:
                raise DataMateError('Embedding token budget cannot fit a source character')
            end = position + low
            chunk_id = disk_index.fingerprint([scope, source, digest, section_number, position, end, chunk_tokens, overlap])
            result.append(Chunk(piece, source, scope, digest,
                                {**(metadata or {}), 'chunk_id': chunk_id, 'section': section_number,
                                 'start_character': position, 'end_character': end}))
            if end == len(value):
                break
            # Retain a token-bounded suffix without re-encoding/rewriting source.
            start = end
            while start > position + 1 and count(value[start - 1:end]) <= overlap:
                start -= 1
            position = max(position + 1, start)
    return result


class NullRetriever:
    def prepare(self, business_ids, *, ingestor, remaining_seconds):
        pass

    def retrieve(self, question, business_id, catalog, remaining_seconds):
        return [], {'mode': 'none', 'status': 'disabled'}

    def identity(self):
        return {'mode': 'none'}

    def runtime_info(self):
        return self.identity()

    def close(self):
        pass


class ContextRetriever:
    def __init__(self, settings, split, *, cache_dir='.cache/datamate', encoder=None):
        self.settings, self.split = settings, split
        self.lock = threading.RLock()
        self._closed = threading.Event()
        self.index = OrderedDict()
        self.embedding_manifest = verify_artifact(settings.embedding_dir, settings.embedding_key)
        self.reranker_manifest = (verify_artifact(settings.reranker_dir, settings.reranker_key)
                                  if settings.mode == 'dense_rerank' else None)
        self.encoder = encoder or VietnameseEncoders(settings)
        self.cache_dir = ensure_outside_dataset(settings.index_dir or Path(cache_dir) / 'retrieval', split.root)
        self._source_hash = disk_index.fingerprint({
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        self._stats = {'prepare_seconds': 0.0, 'cache_hits': 0, 'cache_builds': 0}
        self._prepared = set()

    def identity(self):
        return {'mode': self.settings.mode, 'settings': asdict(self.settings),
                'embedding': self.embedding_manifest, 'reranker': self.reranker_manifest,
                'source_sha256': self._source_hash}

    def runtime_info(self):
        return {**self.identity(), **self._stats, 'closed': self._closed.is_set(),
                'device': self.settings.device, 'dtype': getattr(self.encoder, 'dtype', None),
                'cached_businesses': len(self.index)}

    def _check(self, deadline):
        if self._closed.is_set():
            raise RetrievalFatalError('Retriever is closed')
        if time.monotonic() >= deadline:
            raise RetrievalTimeoutError('RAG deadline exhausted')

    def _corpus(self, business_id, catalog):
        root = self.split.root
        business_root = self.split.business_path(business_id)
        metadata = {'split_fingerprint': catalog.fingerprint}
        result = []
        digests = {}

        def document(text, path, scope, extra):
            if path not in digests:
                digests[path] = sha256(path)
            result.extend(sections(text, path.relative_to(root).as_posix(), scope,
                tokenizer=self.encoder.tokenizer, chunk_tokens=self.settings.chunk_tokens,
                overlap=self.settings.chunk_overlap, source_sha256=digests[path],
                metadata={**metadata, **extra}))

        self.encoder.load()
        document(self.split.rulebook, root / 'rulebook.md', 'split', {'source_kind': 'rulebook'})
        readme_path = business_root / 'README.md'
        if readme_path.is_file():
            document(self.split.readme(business_id), readme_path, business_id, {'source_kind': 'readme'})
        files = sorted(p for p in business_root.rglob('*') if p.is_file()
                       and p.suffix.lower() in {'.csv', '.xlsx', '.sqlite', '.db'})
        for table in catalog.tables:
            if table['name'] == 'calendar':
                continue
            # Ingestor assigns raw_<sorted file index>_<table/sheet>. Its
            # public file field is a basename, so reconstruct nested paths from
            # the same raw-file ordering rather than claiming a description hash.
            match = re.match(r'raw_(\d+)(?:_|$)', table['name'])
            path = files[int(match[1])] if match and int(match[1]) < len(files) else None
            if path is None:
                candidates = [p for p in files if p.name == table.get('file')]
                path = candidates[0] if len(candidates) == 1 else None
            if path is None or not path.resolve().is_relative_to(business_root.resolve()):
                raise RetrievalFatalError('Cannot establish raw-schema source provenance')
            columns = [c['name'] if isinstance(c, dict) else str(c) for c in table.get('columns', [])]
            description = f"Bảng {table['name']} từ {path.relative_to(business_root)}; cột: {', '.join(columns)}"
            document(description, path, business_id, {'source_kind': 'schema', 'table': table['name'],
                                                     'sheet': table.get('sheet'), 'columns': columns})
        return result

    def _index(self, business_id, catalog, deadline):
        key = (business_id, catalog.fingerprint)
        if key in self.index:
            self.index.move_to_end(key)
            return self.index[key]
        self._check(deadline)
        chunks = self._corpus(business_id, catalog)
        self._check(deadline)
        identity = {'business_id': business_id, 'catalog': catalog.fingerprint,
                    'rulebook': self.split.rulebook_hash, 'embedding': self.embedding_manifest,
                    'source_code': self._source_hash,
                    'encoding': {name: getattr(self.settings, name) for name in
                        ('embedding_max_tokens', 'chunk_tokens', 'chunk_overlap', 'dtype', 'device')},
                    'chunks': [c.to_dict() for c in chunks]}
        folder = self.cache_dir / disk_index.fingerprint(identity)
        vectors = disk_index.load(folder, identity, chunks)
        if vectors is None:
            self._check(deadline)
            vectors = validate_vectors(self.encoder.encode([c.text for c in chunks],
                        remaining_seconds=deadline - time.monotonic()), expected_rows=len(chunks))
            self._check(deadline)
            disk_index.save(folder, identity, chunks, vectors)
            self._stats['cache_builds'] += 1
        else:
            self._stats['cache_hits'] += 1
        self._check(deadline)
        self.index[key] = (chunks, vectors)
        while len(self.index) > self.settings.max_cached_businesses:
            self.index.popitem(last=False)
        return chunks, vectors

    def prepare(self, business_ids, *, ingestor, remaining_seconds):
        started = time.monotonic()
        deadline = started + remaining_seconds
        with self.lock:
            try:
                self._check(deadline)
                self.encoder.load()
                for business_id in sorted(set(business_ids)):
                    self._check(deadline)
                    catalog = ingestor.catalog(business_id, normalize=False)
                    self._index(business_id, catalog, deadline)
                    self._prepared.add(business_id)
            except (RetrievalTimeoutError, RetrievalFatalError):
                raise
            except Exception as exc:
                raise RetrievalFatalError(f'RAG preparation failed: {exc}') from exc
            finally:
                self._stats['prepare_seconds'] += time.monotonic() - started

    def retrieve(self, question, business_id, catalog, remaining_seconds):
        start = time.monotonic()
        deadline = start + max(0, remaining_seconds)
        self._check_closed()
        if not self.lock.acquire(timeout=max(0, remaining_seconds)):
            return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'lock'}
        try:
            self._check(deadline)
            chunks, vectors = self._index(business_id, catalog, deadline)
            query = validate_vectors(self.encoder.encode([question],
                remaining_seconds=deadline-time.monotonic()), expected_rows=1)[0]
            self._check(deadline)
            scores = vectors @ query
            candidates = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].metadata['chunk_id']))[:self.settings.candidate_k]
            rerank_scores = {}
            if self.settings.mode == 'dense_rerank' and candidates:
                values = self.encoder.score(question, [chunks[i].text for i in candidates],
                                            remaining_seconds=deadline-time.monotonic())
                if len(values) != len(candidates):
                    raise DataMateError('Reranker alignment mismatch')
                rerank_scores = dict(zip(candidates, values))
                candidates.sort(key=lambda i: (-rerank_scores[i], -float(scores[i]), chunks[i].metadata['chunk_id']))
            self._check(deadline)
            chosen = [{**chunks[i].to_dict(), 'dense_score': float(scores[i]),
                       'rerank_score': rerank_scores.get(i)} for i in candidates[:self.settings.context_k]]
            return chosen, {'mode': self.settings.mode, 'status': 'ok' if chosen else 'empty',
                            'candidate_count': len(candidates), 'seconds': time.monotonic()-start,
                            'index_fingerprint': catalog.fingerprint,
                            'selected_sources': [c['source'] for c in chosen]}
        except RetrievalFatalError:
            raise
        except RetrievalTimeoutError as exc:
            return [], {'mode': self.settings.mode, 'status': 'timeout', 'error': str(exc),
                        'seconds': time.monotonic()-start}
        except (DataMateError, ValueError) as exc:
            return [], {'mode': self.settings.mode, 'status': 'fallback', 'error': str(exc),
                        'seconds': time.monotonic()-start}
        except Exception as exc:
            raise RetrievalFatalError(f'RAG infrastructure failure: {exc}') from exc
        finally:
            self.lock.release()

    def _check_closed(self):
        if self._closed.is_set():
            raise RetrievalFatalError('Retriever is closed')

    def close(self):
        self._closed.set()  # Pending callers cannot reload while close waits.
        with self.lock:
            self.encoder.close()
            self.index.clear()


def build_retriever(settings, split):
    if settings.rag.mode == 'none':
        return NullRetriever()
    return ContextRetriever(settings.rag, split, cache_dir=settings.cache_dir)
