"""Local Vietnamese CLS encoder and one-logit pair reranker adapters."""
import time

from ..domain import DataMateError
from .contracts import RetrievalFatalError, RetrievalTimeoutError


def validate_vectors(values, *, expected_rows):
    import numpy as np
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (expected_rows, 1024) or not np.isfinite(vectors).all():
        raise DataMateError('Embedding requires finite N x 1024 vectors')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise DataMateError('Embedding contains a zero vector')
    return vectors / norms


class VietnameseEncoders:
    def __init__(self, settings):
        self.settings = settings
        self.encoder = self.reranker = self.tokenizer = self.reranker_tokenizer = None
        self.closed = False

    def load(self):
        if self.closed:
            raise RetrievalFatalError('Retriever is closed')
        if self.encoder is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
            dtype = self.settings.dtype
            if dtype == 'auto':
                dtype = 'float16' if self.settings.device.startswith('cuda') else 'float32'
            dtype = getattr(torch, dtype)
            tokenizer = AutoTokenizer.from_pretrained(self.settings.embedding_dir, local_files_only=True, trust_remote_code=False)
            encoder = AutoModel.from_pretrained(self.settings.embedding_dir, local_files_only=True,
                                                trust_remote_code=False, dtype=dtype).to(self.settings.device).eval()
            reranker, reranker_tokenizer = None, None
            if self.settings.mode == 'dense_rerank':
                reranker_tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_dir, local_files_only=True, trust_remote_code=False)
                reranker = AutoModelForSequenceClassification.from_pretrained(
                    self.settings.reranker_dir, local_files_only=True, trust_remote_code=False,
                    dtype=dtype).to(self.settings.device).eval()
        except Exception as exc:
            # Publish neither model on a partial failure; the next explicit
            # initialization cannot silently skip the requested reranker.
            raise RetrievalFatalError(f'Cannot initialize RAG models: {exc}') from exc
        self.torch, self.dtype = torch, str(dtype)
        self.encoder, self.reranker = encoder, reranker
        self.tokenizer, self.reranker_tokenizer = tokenizer, reranker_tokenizer

    def _check(self, deadline):
        if self.closed:
            raise RetrievalFatalError('Retriever is closed')
        if time.monotonic() >= deadline:
            raise RetrievalTimeoutError('RAG batch deadline exhausted')

    def encode(self, texts, *, remaining_seconds):
        import numpy as np
        deadline = time.monotonic() + remaining_seconds
        self._check(deadline)
        self.load()
        vectors = []
        try:
            for start in range(0, len(texts), self.settings.embedding_batch_size):
                self._check(deadline)
                encoded = self.tokenizer(texts[start:start + self.settings.embedding_batch_size], padding=True,
                    truncation=True, max_length=self.settings.embedding_max_tokens, return_tensors='pt').to(self.settings.device)
                with self.torch.inference_mode():
                    hidden = self.encoder(**encoded).last_hidden_state[:, 0, :].float()
                    hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=-1)
                vectors.append(hidden.cpu().numpy())
                self._check(deadline)
        except RuntimeError as exc:
            raise RetrievalFatalError(f'RAG encoder failure: {exc}') from exc
        return validate_vectors(np.concatenate(vectors) if vectors else np.empty((0, 1024)), expected_rows=len(texts))

    def score(self, query, passages, *, remaining_seconds):
        import numpy as np
        deadline = time.monotonic() + remaining_seconds
        self._check(deadline)
        self.load()
        if self.reranker is None:
            raise RetrievalFatalError('Requested reranker was not initialized')
        query_ids = self.reranker_tokenizer.encode(query, add_special_tokens=False)[:self.settings.rerank_query_max_tokens]
        short_query = self.reranker_tokenizer.decode(query_ids, skip_special_tokens=True)
        scores = []
        try:
            for offset in range(0, len(passages), self.settings.rerank_batch_size):
                self._check(deadline)
                batch = passages[offset:offset + self.settings.rerank_batch_size]
                encoded = self.reranker_tokenizer([(short_query, text) for text in batch], padding=True,
                    truncation='only_second', max_length=self.settings.rerank_max_tokens, return_tensors='pt').to(self.settings.device)
                with self.torch.inference_mode():
                    logits = self.reranker(**encoded).logits.float().cpu().numpy()
                if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                    raise DataMateError('Reranker must return one finite logit per passage')
                scores.extend(float(x) for x in logits[:, 0])
                self._check(deadline)
        except RuntimeError as exc:
            raise RetrievalFatalError(f'RAG reranker failure: {exc}') from exc
        return scores

    def close(self):
        self.closed = True
        self.encoder = self.reranker = None
