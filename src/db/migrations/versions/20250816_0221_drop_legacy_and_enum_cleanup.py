"""
Drop legacy columns and cleanup unit_type ENUM

Revision ID: 20250816_0221
Revises: 20250816_0129
Create Date: 2025-08-16 02:21:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250816_0221'
down_revision = '20250816_0129'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        # Safe no-op for non-Postgres
        return

    # 1) Drop indexes on legacy parent_* columns if exist
    op.execute("""
    DO $$ BEGIN
      IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname = 'idx_units_parent_pasal' AND n.nspname = current_schema()
      ) THEN
        EXECUTE 'DROP INDEX IF EXISTS idx_units_parent_pasal';
      END IF;
      IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname = 'idx_units_parent_ayat' AND n.nspname = current_schema()
      ) THEN
        EXECUTE 'DROP INDEX IF EXISTS idx_units_parent_ayat';
      END IF;
      IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname = 'idx_units_parent_huruf' AND n.nspname = current_schema()
      ) THEN
        EXECUTE 'DROP INDEX IF EXISTS idx_units_parent_huruf';
      END IF;
    END $$;
    """)

    # 2) Drop legacy columns if exist: path, parent_pasal_id, parent_ayat_id, parent_huruf_id
    op.execute("""
    DO $$ BEGIN
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='path'
      ) THEN
        ALTER TABLE legal_units DROP COLUMN path;
      END IF;
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_pasal_id'
      ) THEN
        ALTER TABLE legal_units DROP COLUMN parent_pasal_id;
      END IF;
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_ayat_id'
      ) THEN
        ALTER TABLE legal_units DROP COLUMN parent_ayat_id;
      END IF;
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_huruf_id'
      ) THEN
        ALTER TABLE legal_units DROP COLUMN parent_huruf_id;
      END IF;
    END $$;
    """)

    # 3) Clean up ENUM type for unit_type using a temp TEXT column to avoid operator issues
    #    Steps: normalize values -> copy to text -> drop column -> drop type -> recreate type -> add column -> copy back -> drop temp
    op.execute("""
    DO $$
    DECLARE
      old_type regtype;
      old_type_name text;
    BEGIN
      -- Determine current enum type of legal_units.unit_type
      SELECT atttypid::regtype, t.typname
        INTO old_type, old_type_name
      FROM pg_attribute a
      JOIN pg_type t ON t.oid = a.atttypid
      WHERE a.attrelid = 'legal_units'::regclass
        AND a.attname = 'unit_type';

      -- Normalize deprecated rows BEFORE changing type
      EXECUTE 'UPDATE legal_units SET unit_type = ''ANGKA'' WHERE unit_type::text = ''ANGKA_AMANDEMENT''';

      -- Drop default (if any) to reduce dependencies
      EXECUTE 'ALTER TABLE legal_units ALTER COLUMN unit_type DROP DEFAULT';

      -- Create temp column to hold values as text
      EXECUTE 'ALTER TABLE legal_units ADD COLUMN unit_type_tmp text';
      EXECUTE 'UPDATE legal_units SET unit_type_tmp = unit_type::text';

      -- Drop the original enum column (drops dependent operators on that column)
      EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type';

      -- Drop the old enum type
      EXECUTE format('DROP TYPE %s', old_type);

      -- Recreate enum with the SAME NAME but without deprecated value
      EXECUTE format('
        CREATE TYPE %I AS ENUM (
          ''DOKUMEN'', ''BUKU'', ''BAB'', ''BAGIAN'', ''PARAGRAF'',
          ''PASAL'', ''AYAT'', ''HURUF'', ''ANGKA''
        );
      ', old_type_name);

      -- Re-add the column with the enum type
      EXECUTE format('ALTER TABLE legal_units ADD COLUMN unit_type %I', old_type_name);

      -- Copy data back from temp (cast from text to enum)
      EXECUTE format('UPDATE legal_units SET unit_type = unit_type_tmp::%I', old_type_name);

      -- Drop temp column
      EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type_tmp';
    END $$;
    """)


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        return

    # 1) Recreate legacy columns (nullable) and indexes
    op.execute("""
    DO $$ BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='path'
      ) THEN
        ALTER TABLE legal_units ADD COLUMN path jsonb;
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_pasal_id'
      ) THEN
        ALTER TABLE legal_units ADD COLUMN parent_pasal_id varchar(500);
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_ayat_id'
      ) THEN
        ALTER TABLE legal_units ADD COLUMN parent_ayat_id varchar(500);
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='legal_units' AND column_name='parent_huruf_id'
      ) THEN
        ALTER TABLE legal_units ADD COLUMN parent_huruf_id varchar(500);
      END IF;
    END $$;
    """)

    op.execute("""
    CREATE INDEX IF NOT EXISTS idx_units_parent_pasal ON legal_units (parent_pasal_id);
    CREATE INDEX IF NOT EXISTS idx_units_parent_ayat ON legal_units (parent_ayat_id);
    CREATE INDEX IF NOT EXISTS idx_units_parent_huruf ON legal_units (parent_huruf_id);
    """)

    # 2) Reintroduce deprecated enum value by recreating type with it included using temp-column approach
    op.execute("""
    DO $$
    DECLARE
      old_type regtype;
      old_type_name text;
    BEGIN
      SELECT atttypid::regtype, t.typname
        INTO old_type, old_type_name
      FROM pg_attribute a
      JOIN pg_type t ON t.oid = a.atttypid
      WHERE a.attrelid = 'legal_units'::regclass
        AND a.attname = 'unit_type';

      -- Drop default (if any)
      EXECUTE 'ALTER TABLE legal_units ALTER COLUMN unit_type DROP DEFAULT';

      -- Create temp text column and copy values
      EXECUTE 'ALTER TABLE legal_units ADD COLUMN unit_type_tmp text';
      EXECUTE 'UPDATE legal_units SET unit_type_tmp = unit_type::text';

      -- Drop enum column, then drop enum type
      EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type';
      EXECUTE format('DROP TYPE %s', old_type);

      -- Recreate enum including deprecated value
      EXECUTE format('
        CREATE TYPE %I AS ENUM (
          ''DOKUMEN'', ''BUKU'', ''BAB'', ''BAGIAN'', ''PARAGRAF'',
          ''PASAL'', ''AYAT'', ''HURUF'', ''ANGKA'', ''ANGKA_AMANDEMENT''
        );
      ', old_type_name);

      -- Re-add enum column and copy back
      EXECUTE format('ALTER TABLE legal_units ADD COLUMN unit_type %I', old_type_name);
      EXECUTE format('UPDATE legal_units SET unit_type = unit_type_tmp::%I', old_type_name);

      -- Drop temp column
      EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type_tmp';
    END $$;
    """)
