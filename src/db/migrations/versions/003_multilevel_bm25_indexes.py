"""Multi-level BM25 indexes for PASAL content + granular bm25_body

Revision ID: 003_multilevel_bm25_indexes
Revises: 002_add_fts_bm25_search
Create Date: 2024-01-15 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


# revision identifiers, used by Alembic.
revision = '003_multilevel_bm25_indexes'
down_revision = '002_add_fts_bm25_search'
branch_labels = None
depends_on = None


def upgrade():
    """Create multi-level BM25 indexes: PASAL content + granular bm25_body"""

    # Create GIN index for PASAL content field FTS
    op.execute("""
        CREATE INDEX idx_pasal_content_fts
        ON legal_units USING gin(to_tsvector('indonesian', content))
        WHERE unit_type = 'PASAL' AND content IS NOT NULL
    """)

    # Create GIN index for granular level bm25_body FTS (AYAT/HURUF/ANGKA)
    # Note: This enhances the existing bm25_tsvector index with proper filtering
    op.execute("""
        CREATE INDEX idx_granular_bm25_fts
        ON legal_units USING gin(bm25_tsvector)
        WHERE unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_body IS NOT NULL
    """)

    # Create composite index for multi-level search performance
    op.execute("""
        CREATE INDEX idx_multilevel_search_performance
        ON legal_units (unit_type, document_id)
        WHERE (unit_type = 'PASAL' AND content IS NOT NULL)
           OR (unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_body IS NOT NULL)
    """)

    # Create function to update bm25_tsvector with Indonesian language
    op.execute("""
        CREATE OR REPLACE FUNCTION update_bm25_tsvector_indonesian()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Update tsvector with Indonesian configuration for granular units
            IF NEW.unit_type IN ('AYAT', 'HURUF', 'ANGKA') THEN
                NEW.bm25_tsvector = to_tsvector('indonesian', COALESCE(NEW.bm25_body, ''));
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Update existing trigger to use Indonesian language
    op.execute("DROP TRIGGER IF EXISTS trigger_update_bm25_tsvector ON legal_units;")

    op.execute("""
        CREATE TRIGGER trigger_update_bm25_tsvector_indonesian
            BEFORE INSERT OR UPDATE OF bm25_body ON legal_units
            FOR EACH ROW
            EXECUTE FUNCTION update_bm25_tsvector_indonesian();
    """)

    # Update existing bm25_tsvector data to use Indonesian language
    op.execute("""
        UPDATE legal_units
        SET bm25_tsvector = to_tsvector('indonesian', COALESCE(bm25_body, ''))
        WHERE unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_body IS NOT NULL;
    """)

    # Create additional performance indexes
    op.execute("""
        CREATE INDEX idx_pasal_units_doc_status
        ON legal_units (document_id)
        WHERE unit_type = 'PASAL' AND content IS NOT NULL
    """)

    op.execute("""
        CREATE INDEX idx_granular_units_doc_status
        ON legal_units (document_id)
        WHERE unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_body IS NOT NULL
    """)


def downgrade():
    """Remove multi-level BM25 indexes and revert to simple configuration"""

    # Drop multi-level indexes
    op.execute("DROP INDEX IF EXISTS idx_pasal_content_fts;")
    op.execute("DROP INDEX IF EXISTS idx_granular_bm25_fts;")
    op.execute("DROP INDEX IF EXISTS idx_multilevel_search_performance;")
    op.execute("DROP INDEX IF EXISTS idx_pasal_units_doc_status;")
    op.execute("DROP INDEX IF EXISTS idx_granular_units_doc_status;")

    # Drop Indonesian trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_bm25_tsvector_indonesian ON legal_units;")
    op.execute("DROP FUNCTION IF EXISTS update_bm25_tsvector_indonesian();")

    # Restore original simple trigger
    op.execute("""
        CREATE TRIGGER trigger_update_bm25_tsvector
            BEFORE INSERT OR UPDATE OF bm25_body ON legal_units
            FOR EACH ROW
            EXECUTE FUNCTION update_bm25_tsvector();
    """)

    # Revert bm25_tsvector to simple configuration
    op.execute("""
        UPDATE legal_units
        SET bm25_tsvector = to_tsvector('simple', COALESCE(bm25_body, ''))
        WHERE bm25_body IS NOT NULL;
    """)
