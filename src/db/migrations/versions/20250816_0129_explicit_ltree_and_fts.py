"""
Explicit retrieval foundations: ltree + FTS for legal_units

Revision ID: 20250816_0129
Revises: 
Create Date: 2025-08-16 01:29:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250816_0129'
down_revision = '003_multilevel_bm25_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    dialect = conn.dialect.name

    if dialect != 'postgresql':
        # No-op on non-Postgres backends
        return

    # 1) Ensure extensions
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS ltree"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS unaccent"))

    # 2) Add columns if not exist
    op.execute("""
    ALTER TABLE legal_units
      ADD COLUMN IF NOT EXISTS unit_path ltree,
      ADD COLUMN IF NOT EXISTS parent_unit_id UUID
    """)

    op.execute("""
    ALTER TABLE legal_units
      ADD COLUMN IF NOT EXISTS tsv_content tsvector
    """)

    # 3) Add FK if not exists (self-referential to primary key id)
    op.execute(
        """
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'legal_units_parent_fk'
          ) THEN
            ALTER TABLE legal_units
              ADD CONSTRAINT legal_units_parent_fk
              FOREIGN KEY (parent_unit_id) REFERENCES legal_units(id)
              ON DELETE SET NULL;
          END IF;
        END $$;
        """
    )

    # 4) Indexes
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS legal_units_unit_path_gist
          ON legal_units USING GIST (unit_path);
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS legal_units_tsv_content_gin
          ON legal_units USING GIN (tsv_content);
        """
    )

    # Helpful index for parent lookup
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_legal_units_parent_unit_id
          ON legal_units (parent_unit_id);
        """
    )

    # 5) Trigger for tsv_content maintenance
    op.execute(
        """
        DO $do$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = 'legal_units_tsvector_update'
              AND p.pronargs = 0
              AND n.nspname = current_schema()
          ) THEN
            EXECUTE $sql$
              CREATE FUNCTION legal_units_tsvector_update() RETURNS trigger AS $fn$
              BEGIN
                NEW.tsv_content :=
                  setweight(to_tsvector('indonesian', coalesce(NEW.title, '')), 'A') ||
                  setweight(to_tsvector('indonesian', coalesce(NEW.display_text, '')), 'B') ||
                  setweight(to_tsvector('indonesian', coalesce(NEW.content, '')), 'C');
                RETURN NEW;
              END
              $fn$ LANGUAGE plpgsql;
            $sql$;
          END IF;
        END
        $do$;
        """
    )

    op.execute(
        """
        DO $do$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE t.tgname = 'legal_units_tsvector_update'
              AND c.relname = 'legal_units'
              AND n.nspname = current_schema()
              AND NOT t.tgisinternal
          ) THEN
            EXECUTE $sql$
              CREATE TRIGGER legal_units_tsvector_update
              BEFORE INSERT OR UPDATE OF title, display_text, content ON legal_units
              FOR EACH ROW EXECUTE FUNCTION legal_units_tsvector_update();
            $sql$;
          END IF;
        END
        $do$;
        """
    )

    # 6) One-time backfill to populate existing rows
    op.execute(
        """
        UPDATE legal_units
        SET tsv_content =
          setweight(to_tsvector('indonesian', coalesce(title, '')), 'A') ||
          setweight(to_tsvector('indonesian', coalesce(display_text, '')), 'B') ||
          setweight(to_tsvector('indonesian', coalesce(content, '')), 'C')
        WHERE tsv_content IS NULL;
        """
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        return
    # Drop trigger and function
    op.execute(
        """
        DO $do$ BEGIN
          IF EXISTS (
            SELECT 1
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE t.tgname = 'legal_units_tsvector_update'
              AND c.relname = 'legal_units'
              AND n.nspname = current_schema()
              AND NOT t.tgisinternal
          ) THEN
            EXECUTE 'DROP TRIGGER legal_units_tsvector_update ON legal_units';
          END IF;
        END $do$;
        """
    )
    op.execute(
        """
        DO $do$ BEGIN
          IF EXISTS (
            SELECT 1
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = 'legal_units_tsvector_update'
              AND p.pronargs = 0
              AND n.nspname = current_schema()
          ) THEN
            EXECUTE 'DROP FUNCTION legal_units_tsvector_update()';
          END IF;
        END $do$;
        """
    )
    # Keep columns and extensions (safe rollback minimal)
