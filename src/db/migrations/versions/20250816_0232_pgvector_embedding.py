"""
Add pgvector extension, embedding column, and ANN index for legal_units

Revision ID: 20250816_0232
Revises: 20250816_0221
Create Date: 2025-08-16 02:32:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250816_0232"
down_revision = "20250816_0221"
branch_labels = None
depends_on = None


EMBEDDING_DIM = 768  # adjust if your embedding model differs


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    # 1) Ensure pgvector extension
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2) Add embedding column if not exists
    op.execute(
        sa.text(
            f"""
            DO $$ BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='embedding'
              ) THEN
                ALTER TABLE legal_units ADD COLUMN embedding vector({EMBEDDING_DIM});
              END IF;
            END $$;
            """
        )
    )

    # 3) Create ANN index (prefer HNSW when available, else IVFFLAT)
    #    Detect availability of HNSW access method; fall back to IVFFLAT otherwise
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM pg_am WHERE amname = 'hnsw') THEN
                -- HNSW available (pgvector >= 0.6.0)
                EXECUTE 'CREATE INDEX IF NOT EXISTS legal_units_embedding_hnsw
                         ON legal_units USING hnsw (embedding vector_cosine)
                         WITH (m=16, ef_construction=64)';
              ELSE
                -- Fallback: IVFFLAT
                EXECUTE 'CREATE INDEX IF NOT EXISTS legal_units_embedding_ivfflat
                         ON legal_units USING ivfflat (embedding vector_cosine)
                         WITH (lists=100)';
              END IF;
            EXCEPTION WHEN undefined_object THEN
              -- Very old pgvector without hnsw or ivfflat registered; skip index creation safely
              NULL;
            END
            $$;
            """
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    # Drop indexes if exist (both variants)
    op.execute(
        """
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE c.relname='legal_units_embedding_hnsw' AND n.nspname=current_schema()
          ) THEN
            EXECUTE 'DROP INDEX IF EXISTS legal_units_embedding_hnsw';
          END IF;
          IF EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE c.relname='legal_units_embedding_ivfflat' AND n.nspname=current_schema()
          ) THEN
            EXECUTE 'DROP INDEX IF EXISTS legal_units_embedding_ivfflat';
          END IF;
        END $$;
        """
    )

    # Drop column (non-destructive for schema; data loss acceptable on downgrade)
    op.execute(
        """
        DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='legal_units' AND column_name='embedding'
          ) THEN
            ALTER TABLE legal_units DROP COLUMN embedding;
          END IF;
        END $$;
        """
    )

    # Do not drop extension automatically on downgrade (other objects may depend on it)
