"""add parent_ayat_id and parent_huruf_id to legal_units

Revision ID: ea3062318da4
Revises: 
Create Date: 2025-08-10 06:50:10.228988

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ea3062318da4'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Add new hierarchy columns
    op.add_column(
        'legal_units',
        sa.Column('parent_ayat_id', sa.String(length=500), nullable=True)
    )
    op.add_column(
        'legal_units',
        sa.Column('parent_huruf_id', sa.String(length=500), nullable=True)
    )

    # Create indexes to support fast hierarchical lookups
    op.create_index('idx_units_parent_pasal', 'legal_units', ['parent_pasal_id'], unique=False)
    op.create_index('idx_units_parent_ayat', 'legal_units', ['parent_ayat_id'], unique=False)
    op.create_index('idx_units_parent_huruf', 'legal_units', ['parent_huruf_id'], unique=False)

    # NOTE: Backfill should be done via ingestion re-run or a separate one-off script
    # that traverses existing hierarchy to populate parent_ayat_id and parent_huruf_id.
    # Leaving data migration out of schema migration to keep it reversible and safe.


def downgrade() -> None:
    """Downgrade database schema."""
    # Drop indexes first
    op.drop_index('idx_units_parent_huruf', table_name='legal_units')
    op.drop_index('idx_units_parent_ayat', table_name='legal_units')
    op.drop_index('idx_units_parent_pasal', table_name='legal_units')

    # Drop columns
    op.drop_column('legal_units', 'parent_huruf_id')
    op.drop_column('legal_units', 'parent_ayat_id')
