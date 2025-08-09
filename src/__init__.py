# src package initialization
"""
Legal Database System - Source Package
Main package for Indonesian legal document processing system.
"""

__version__ = "1.0.0"
__author__ = "Legal Database Refactoring Team"
__description__ = "Indonesian Legal Document Processing and RAG System"

# Make key modules easily importable
from . import config
from . import services
from . import utils

__all__ = [
    'config',
    'services',
    'utils'
]
