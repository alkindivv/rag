"""Add FTS BM25 search enhancement

Revision ID: 002_add_fts_bm25_search
Revises: 001_remove_fts_ordinal_dense_search
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


# revision identifiers, used by Alembic.
revision = '002_add_fts_bm25_search'
down_revision = '001_dense_search'
branch_labels = None
depends_on = None


def upgrade():
    """Add FTS enhancement for BM25 search"""

    # Add tsvector column for FTS
    op.add_column('legal_units',
        sa.Column('bm25_tsvector', TSVECTOR, nullable=True)
    )

    # Create GIN index for efficient FTS search
    op.execute("""
        CREATE INDEX idx_legal_units_bm25_tsvector_gin
        ON legal_units USING gin(bm25_tsvector)
    """)

    # Create function to update tsvector automatically
    op.execute("""
        CREATE OR REPLACE FUNCTION update_bm25_tsvector()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update tsvector with Indonesian configuration
            NEW.bm25_tsvector = to_tsvector('simple', COALESCE(NEW.bm25_body, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create trigger to automatically update tsvector
    op.execute("""
        CREATE TRIGGER trigger_update_bm25_tsvector
            BEFORE INSERT OR UPDATE OF bm25_body ON legal_units
            FOR EACH ROW
            EXECUTE FUNCTION update_bm25_tsvector();
    """)

    # Populate existing data
    op.execute("""
        UPDATE legal_units
        SET bm25_tsvector = to_tsvector('simple', COALESCE(bm25_body, ''))
        WHERE bm25_body IS NOT NULL;
    """)

    # Add index on unit_type for BM25 queries (performance optimization)
    op.execute("""
        CREATE INDEX idx_legal_units_unit_type_bm25
        ON legal_units (unit_type)
        WHERE bm25_body IS NOT NULL;
    """)


def downgrade():
    """Remove FTS enhancement"""

    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS trigger_update_bm25_tsvector ON legal_units;")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_bm25_tsvector();")

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_legal_units_bm25_tsvector_gin;")
    op.execute("DROP INDEX IF EXISTS idx_legal_units_unit_type_bm25;")

    # Drop column
    op.drop_column('legal_units', 'bm25_tsvector')
