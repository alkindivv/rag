"""
Models Package
Clean, production-ready models for legal document storage and vector search.

Simple, no overengineering - just what works.
"""

# Document storage models
from .document_storage import LegalDocument, Base as DocumentBase

# Vector storage models
from .vector_storage import DocumentVector, VectorSearchLog, Base as VectorBase

__all__ = [
    # Document storage
    'LegalDocument',
    'DocumentBase',

    # Vector storage
    'DocumentVector',
    'VectorSearchLog',
    'VectorBase'
]
