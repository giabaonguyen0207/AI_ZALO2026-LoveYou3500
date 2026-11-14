"""Lightweight retrieval errors; importing this module never loads models."""
from ..domain import DataMateError


class RetrievalFatalError(DataMateError):
    """Model/index initialization or infrastructure cannot safely recover."""


class RetrievalTimeoutError(DataMateError):
    """The current retrieval exhausted its allotted time."""
