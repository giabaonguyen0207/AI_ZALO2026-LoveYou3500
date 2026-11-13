"""Local, source-scoped retrieval over rulebook, README and raw schema.

Retrieved text is evidence for planning, never a substitute for SQL over rows.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass

from ..domain import DataMateError
from ..models.artifacts import verify_artifact
from ..models.artifacts import sha256
from ..data.transforms import search_key


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    scope: str
    digest: str

    def to_dict(self):
        return {'text': self.text, 'source': self.source, 'scope': self.scope, 'source_sha256': self.digest}


def sections(text: str, source: str, scope: str) -> list[Chunk]:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    result, current, heading = [], [], ''
    for line in text.splitlines():
        if line.lstrip().startswith('#') and current:
            content = '\n'.join(current).strip()
            if content:
                result.append(Chunk(content, source, scope, digest))
            current = []
        if line.lstrip().startswith('#'):
            heading = line
        if current and sum(len(item) for item in current) + len(line) > 1800:
            result.append(Chunk('\n'.join(current).strip(), source, scope, digest))
            current = [heading] if heading else []
        current.append(line)
    if current:
        content = '\n'.join(current).strip()
        if content:
            result.append(Chunk(content, source, scope, digest))
    return result


class NullRetriever:
    def retrieve(self, question, business_id, catalog, remaining_seconds):
        return [], {'mode': 'none', 'status': 'disabled'}

    def close(self):
        pass


class ContextRetriever:
    def __init__(self, settings, split):
        self.settings, self.split = settings, split
        self.encoder = self.reranker = None
        self.index = {}
        self.lock = threading.RLock()
        self.embedding_manifest = verify_artifact(settings.embedding_dir, settings.embedding_key)
        self.reranker_manifest = (verify_artifact(settings.reranker_dir, settings.reranker_key)
                                  if settings.mode == 'dense_rerank' else None)

    def _load(self):
        if self.encoder is not None:
            return
        # No hub identifiers or remote code reach model loaders at inference.
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.torch = torch
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_dir, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(self.settings.embedding_dir, local_files_only=True, trust_remote_code=False).to(self.settings.device).eval()
        if self.reranker_manifest:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_dir, local_files_only=True, trust_remote_code=False)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                self.settings.reranker_dir, local_files_only=True, trust_remote_code=False).to(self.settings.device).eval()

    def _encode(self, texts):
        self._load()
        vectors = []
        for start in range(0, len(texts), 8):
            encoded = self.encoder_tokenizer(texts[start:start + 8], padding=True, truncation=True,
                                             max_length=2048, return_tensors='pt').to(self.settings.device)
            with self.torch.inference_mode():
                hidden = self.encoder(**encoded).last_hidden_state[:, 0, :].float()
                hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=-1)
            if hidden.shape[1] != 1024 or not self.torch.isfinite(hidden).all():
                raise DataMateError('Embedding shape/values invalid')
            vectors.extend(hidden.cpu().tolist())
        return vectors

    def _corpus(self, business_id, catalog):
        readme = self.split.readme(business_id)
        chunks = sections(self.split.rulebook, 'rulebook.md', 'split')
        chunks += sections(readme, f'data/{business_id}/README.md', business_id)
        # Catalog is built from raw files; no generated binding, answer or trace is indexed.
        digests = {}
        for table in catalog.tables:
            if table['name'] == 'calendar':
                continue
            source = str(table.get('file') or table.get('source') or table['name'])
            columns = table.get('columns', [])
            names = [c['name'] if isinstance(c, dict) else str(c) for c in columns]
            description = f"Bảng {table['name']} từ {source}; cột: {', '.join(names)}"
            path = (self.split.business_path(business_id) / source).resolve()
            root = self.split.business_path(business_id).resolve()
            if source not in digests:
                digests[source] = sha256(path) if path.is_relative_to(root) and path.is_file() else hashlib.sha256(description.encode('utf-8')).hexdigest()
            chunks.append(Chunk(description, f'data/{business_id}/{source}', business_id, digests[source]))
        return chunks

    @staticmethod
    def _question_specific_boost(question, chunk):
        """Tie-break near-equal chunks using facts explicitly present in the query."""
        text = search_key(chunk.text)
        raw = chunk.text.lower()
        q = search_key(question)
        boost = 0.0
        for month, year in re.findall(r'(?<!\d)(\d{1,2})\s*/\s*(\d{4})(?!\d)', question):
            marker = rf'(?:t|thang|month)[ _-]*0?{int(month)}[ _/-]*{year}'
            if re.search(marker, raw):
                boost += 0.35
        if 'huy' in q and any(term in text for term in ('ty le don huy', 'don bi huy', 'trang thai huy')):
            boost += 0.20
        if ('doanh thu' in q or ('ban' in q and 'tien' in q)) and any(term in text for term in ('doanh thu', 'thanh tien', 'line_amount')):
            boost += 0.20
        return boost

    def retrieve(self, question, business_id, catalog, remaining_seconds):
        # Workers share the encoder and index; avoid duplicate initialization.
        started = time.monotonic()
        if not self.lock.acquire(timeout=max(0, remaining_seconds)):
            return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'lock'}
        try:
            return self._retrieve_locked(question, business_id, catalog,
                                         remaining_seconds - (time.monotonic() - started))
        finally:
            self.lock.release()

    def _retrieve_locked(self, question, business_id, catalog, remaining_seconds):
        start = time.monotonic()
        if remaining_seconds <= 0:
            return [], {'mode': self.settings.mode, 'status': 'timeout'}
        try:
            key = (business_id, catalog.fingerprint, self.split.rulebook_hash)
            if key not in self.index:
                chunks = self._corpus(business_id, catalog)
                self.index[key] = (chunks, [])
            chunks, vectors = self.index[key]
            # Persist progress in memory when a CPU index takes longer than
            # one question budget. Subsequent questions resume at this chunk.
            while len(vectors) < len(chunks):
                if time.monotonic() - start >= remaining_seconds:
                    return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'index',
                                'indexed_chunks': len(vectors), 'total_chunks': len(chunks)}
                batch = chunks[len(vectors):len(vectors) + 4]
                vectors.extend(self._encode([c.text for c in batch]))
            if time.monotonic() - start >= remaining_seconds:
                return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'index',
                            'indexed_chunks': len(vectors), 'total_chunks': len(chunks)}
            query = self._encode([question])[0]
            scores = [sum(a*b for a, b in zip(query, vector)) for vector in vectors]
            retrieval_scores = [score + self._question_specific_boost(question, chunk)
                                for score, chunk in zip(scores, chunks)]
            candidates = sorted(range(len(chunks)), key=lambda i: (-retrieval_scores[i], chunks[i].source, i))[:self.settings.candidate_k]
            # Keep a directly relevant split/business-document chunk in the
            # reranker pool.  Otherwise several near-identical workbook-sheet
            # schemas can crowd out the rule that defines the requested metric.
            for scope in ('split', business_id):
                scoped = [i for i, chunk in enumerate(chunks)
                          if chunk.scope == scope and self._question_specific_boost(question, chunk) > 0]
                if not scoped:
                    continue
                best = max(scoped, key=lambda i: (retrieval_scores[i], -i))
                if best not in candidates:
                    candidates[-1] = best
                    candidates = list(dict.fromkeys(candidates))
            if self.reranker is not None and candidates:
                query_ids = self.reranker_tokenizer(question, add_special_tokens=False, truncation=True, max_length=256)['input_ids']
                short_query = self.reranker_tokenizer.decode(query_ids, skip_special_tokens=True)
                pairs = [(short_query, chunks[i].text) for i in candidates]
                reranked = []
                for offset in range(0, len(pairs), 4):
                    if time.monotonic() - start >= remaining_seconds:
                        return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'rerank',
                                    'scored_candidates': len(reranked)}
                    encoded = self.reranker_tokenizer(pairs[offset:offset + 4], padding=True, truncation='only_second',
                                                       max_length=2304, return_tensors='pt').to(self.settings.device)
                    with self.torch.inference_mode():
                        logits = self.reranker(**encoded).logits.view(-1).float().cpu().tolist()
                    reranked.extend(logits)
                rerank_scores = dict(zip(candidates, reranked))
                candidates.sort(key=lambda i: (-rerank_scores[i], -retrieval_scores[i], i))
            chosen = [chunks[i].to_dict() for i in candidates[:self.settings.context_k]]
            if time.monotonic() - start >= remaining_seconds:
                return [], {'mode': self.settings.mode, 'status': 'timeout', 'phase': 'search'}
            return chosen, {'mode': self.settings.mode, 'status': 'ok', 'candidate_count': len(candidates),
                            'selected_sources': [c['source'] for c in chosen],
                            'seconds': round(time.monotonic() - start, 4)}
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                raise
            return [], {'mode': self.settings.mode, 'status': 'fallback', 'error': str(exc)[:250]}
        except (DataMateError, ValueError) as exc:
            return [], {'mode': self.settings.mode, 'status': 'fallback', 'error': str(exc)[:250],
                        'seconds': round(time.monotonic() - start, 4)}

    def close(self):
        self.encoder = self.reranker = None
        self.index.clear()


def build_retriever(settings, split):
    if settings.rag.mode == 'none':
        return NullRetriever()
    return ContextRetriever(settings.rag, split)
