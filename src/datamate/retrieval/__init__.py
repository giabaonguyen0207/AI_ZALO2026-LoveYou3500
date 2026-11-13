"""Question-scoped evidence retrieval; importing this package loads no weights."""

from .core import ContextRetriever, NullRetriever, build_retriever

__all__ = ['ContextRetriever', 'NullRetriever', 'build_retriever']
