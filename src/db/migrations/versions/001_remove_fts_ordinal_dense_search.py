"""remove_fts_ordinal_dense_search

Revision ID: 001_dense_search
Revises: ea3062318da4
Create Date: 2025-01-27 10:00:00.000000

Remove FTS and ordinal fields for dense semantic search only implementation.

Changes:
- Remove content_vector (TSVECTOR) from legal_units
- Remove ordinal_int, ordinal_suffix, seq_sort_key from legal_units
- Drop FTS-related indexes
- Update DocumentVector embedding dimension to 384
- Update HNSW index parameters for better performance
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '001_dense_search'
down_revision = 'ea3062318da4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade to dense search only - remove FTS and ordinal fields."""

    # Drop dependent views first (they depend on ordinal columns)
    op.execute("DROP VIEW IF EXISTS v_search_performance CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_pasal_search CASCADE")

    # Drop FTS and ordinal indexes first
    op.drop_index('idx_units_content_vector_gin', table_name='legal_units', if_exists=True)
    op.drop_index('idx_units_type_ord', table_name='legal_units', if_exists=True)

    # Remove FTS and ordinal columns from legal_units
    op.drop_column('legal_units', 'content_vector')
    op.drop_column('legal_units', 'ordinal_int')
    op.drop_column('legal_units', 'ordinal_suffix')
    op.drop_column('legal_units', 'seq_sort_key')

    # Update DocumentVector table for 384-dimensional embeddings
    # First drop existing HNSW index
    op.drop_index('idx_vec_embedding_hnsw', table_name='document_vectors')

    # Change embedding column dimension
    op.alter_column('document_vectors', 'embedding',
                    existing_type=Vector(1024),
                    type_=Vector(384),
                    existing_nullable=False)

    # Recreate HNSW index with improved parameters
    op.create_index(
        'idx_vec_embedding_hnsw',
        'document_vectors',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 200},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )

    # Add index on content_type for efficient pasal filtering
    op.create_index('idx_vec_content_type', 'document_vectors', ['content_type'])


def downgrade() -> None:
    """Downgrade - restore FTS and ordinal fields."""

    # Drop new index
    op.drop_index('idx_vec_content_type', table_name='document_vectors', if_exists=True)

    # Restore DocumentVector embedding dimension
    op.drop_index('idx_vec_embedding_hnsw', table_name='document_vectors')

    op.alter_column('document_vectors', 'embedding',
                    existing_type=Vector(384),
                    type_=Vector(1024),
                    existing_nullable=False)

    # Recreate old HNSW index
    op.create_index(
        'idx_vec_embedding_hnsw',
        'document_vectors',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )

    # Add back FTS and ordinal columns to legal_units
    op.add_column('legal_units', sa.Column('content_vector', postgresql.TSVECTOR(), nullable=True))
    op.add_column('legal_units', sa.Column('ordinal_int', sa.Integer(), nullable=True, default=0))
    op.add_column('legal_units', sa.Column('ordinal_suffix', sa.String(length=10), nullable=True, default=""))
    op.add_column('legal_units', sa.Column('seq_sort_key', sa.String(length=50), nullable=True))

    # Recreate FTS and ordinal indexes
    op.create_index('idx_units_content_vector_gin', 'legal_units', ['content_vector'],
                    unique=False, postgresql_using='gin')
    op.create_index('idx_units_type_ord', 'legal_units', ['unit_type', 'ordinal_int'], unique=False)
    op.create_index('idx_seq_sort_key', 'legal_units', ['seq_sort_key'], unique=False)

    # Recreate materialized views that were dropped (basic versions)
    op.execute("""
        CREATE MATERIALIZED VIEW mv_pasal_search AS
        SELECT
            lu.id,
            lu.unit_id,
            lu.unit_type,
            lu.number_label,
            lu.ordinal_int,
            lu.content,
            lu.citation_string,
            ld.doc_form,
            ld.doc_year,
            ld.doc_number
        FROM legal_units lu
        JOIN legal_documents ld ON ld.id = lu.document_id
        WHERE lu.unit_type = 'PASAL'
        AND ld.doc_status = 'BERLAKU'
    """)

    op.execute("""
        CREATE VIEW v_search_performance AS
        SELECT
            COUNT(*) as total_pasal,
            COUNT(DISTINCT doc_form) as doc_forms,
            AVG(LENGTH(content)) as avg_content_length
        FROM mv_pasal_search
    """)
