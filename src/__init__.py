# src package initialization
"""
Legal Database System - Source Package
Main package for Indonesian legal document processing system.
"""

__version__ = "1.0.0"
__author__ = "Legal Database Refactoring Team"
__description__ = "Indonesian Legal Document Processing and RAG System"

# Expose common subpackages without importing heavy dependencies at
# import-time. This keeps optional requirements (e.g. SQLAlchemy) from
# being loaded when unused, which is important for lightweight unit
# tests in this kata.

__all__ = ["config", "models", "services", "utils"]
