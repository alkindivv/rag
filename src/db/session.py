"""Database session management for Legal RAG system."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..config.settings import settings


# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Yields:
        Session: SQLAlchemy database session

    Example:
        with get_db_session() as db:
            # Use db session here
            result = db.query(LegalDocument).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables.

    This creates all tables defined in the models and applies required
    Postgres extensions, column types, triggers, and indexes without Alembic.
    """
    from .models import Base, setup_db_extras

    # Create tables from ORM
    Base.metadata.create_all(bind=engine)

    # Apply DB extras (extensions, triggers, indexes, cleanup) centralised in models
    setup_db_extras(engine)


def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data!
    Only use for testing or development.
    """
    from .models import Base

    # First, drop legacy tables not present in ORM that may hold FKs to our tables
    if engine.url.get_backend_name() == "postgresql":
        with engine.begin() as conn:
            # Drop known legacy table to avoid FK dependency errors
            conn.execute(text("DROP TABLE IF EXISTS document_vectors CASCADE"))

    # Then drop all ORM-declared tables
    Base.metadata.drop_all(bind=engine)


def reset_db() -> None:
    """
    Drop and recreate all database tables.

    WARNING: This will delete all data!
    Only use for testing or development.
    """
    drop_db()
    init_db()
