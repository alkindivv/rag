"""
Backfill unit_path (ltree) using recursive CTE based on parent_unit_id and tokenization rules
aligned with orchestrator conventions.

Revision ID: 20250816_0415
Revises: 20250816_0129
Create Date: 2025-08-16 04:15:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250816_0415"
down_revision = "20250816_0232"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    # Helpers: slugify and roman_to_int
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION slugify(txt text)
            RETURNS text AS $$
            DECLARE s text;
            BEGIN
              s := lower(coalesce(txt, ''));
              s := regexp_replace(s, '[-\s]+', '_', 'g');
              s := regexp_replace(s, '[^a-z0-9_]', '', 'g');
              RETURN s;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """
        )
    )

    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION roman_to_int(roman text)
            RETURNS int AS $$
            DECLARE total int := 0;
            DECLARE prev int := 0;
            DECLARE ch char;
            DECLARE val int;
            DECLARE i int;
            BEGIN
              IF roman IS NULL THEN RETURN 0; END IF;
              roman := upper(roman);
              FOR i IN REVERSE length(roman)..1 LOOP
                ch := substr(roman, i, 1);
                val := CASE ch
                  WHEN 'I' THEN 1
                  WHEN 'V' THEN 5
                  WHEN 'X' THEN 10
                  WHEN 'L' THEN 50
                  WHEN 'C' THEN 100
                  WHEN 'D' THEN 500
                  WHEN 'M' THEN 1000
                  ELSE 0
                END;
                IF val < prev THEN
                  total := total - val;
                ELSE
                  total := total + val;
                  prev := val;
                END IF;
              END LOOP;
              RETURN total;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """
        )
    )

    # Build tokens per unit and recurse from roots to leaves to construct full path per row
    op.execute(
        sa.text(
            """
            WITH RECURSIVE tokens AS (
              SELECT
                lu.id,
                lu.parent_unit_id,
                lu.document_id,
                CASE lu.unit_type::text
                  WHEN 'BAB' THEN 'bab_' || (
                    CASE WHEN lu.number_label ~ '^[IVXLCDM]+$'
                         THEN roman_to_int(lu.number_label)::text
                         ELSE slugify(lu.number_label)
                    END)
                  WHEN 'PASAL' THEN 'pasal_' || slugify(lu.number_label)
                  WHEN 'AYAT' THEN 'ayat_' || slugify(lu.number_label)
                  WHEN 'HURUF' THEN 'huruf_' || slugify(lu.number_label)
                  WHEN 'ANGKA' THEN 'angka_' || slugify(lu.number_label)
                  WHEN 'BAGIAN' THEN 'bagian_' || slugify(lu.number_label)
                  WHEN 'PARAGRAF' THEN 'paragraf_' || slugify(lu.number_label)
                  ELSE slugify(lu.unit_type::text)
                END AS token
              FROM legal_units lu
            ),
            roots AS (
              SELECT t.id, t.parent_unit_id, t.document_id, t.token, t.token::text AS path
              FROM tokens t
              WHERE t.parent_unit_id IS NULL
            ),
            rec AS (
              SELECT * FROM roots
              UNION ALL
              SELECT c.id, c.parent_unit_id, c.document_id, c.token, r.path || '.' || c.token AS path
              FROM tokens c
              JOIN rec r ON c.parent_unit_id = r.id
            ),
            doc_root AS (
              SELECT d.id AS document_id, slugify(d.doc_id)::text AS root_token
              FROM legal_documents d
            ),
            final AS (
              SELECT r.id, (dr.root_token || '.' || r.path) AS full_path
              FROM rec r
              JOIN doc_root dr ON dr.document_id = r.document_id
            )
            UPDATE legal_units lu
            SET unit_path = text2ltree(f.full_path)
            FROM final f
            WHERE lu.id = f.id
              AND (lu.unit_path IS NULL OR lu.unit_path::text <> f.full_path);
            """
        )
    )

    # Fallback for orphans (rows with missing parent chain): use doc root + own token
    op.execute(
        sa.text(
            """
            WITH RECURSIVE tokens AS (
              SELECT
                lu.id,
                lu.document_id,
                CASE lu.unit_type::text
                  WHEN 'BAB' THEN 'bab_' || (
                    CASE WHEN lu.number_label ~ '^[IVXLCDM]+$'
                         THEN roman_to_int(lu.number_label)::text
                         ELSE slugify(lu.number_label)
                    END)
                  WHEN 'PASAL' THEN 'pasal_' || slugify(lu.number_label)
                  WHEN 'AYAT' THEN 'ayat_' || slugify(lu.number_label)
                  WHEN 'HURUF' THEN 'huruf_' || slugify(lu.number_label)
                  WHEN 'ANGKA' THEN 'angka_' || slugify(lu.number_label)
                  WHEN 'BAGIAN' THEN 'bagian_' || slugify(lu.number_label)
                  WHEN 'PARAGRAF' THEN 'paragraf_' || slugify(lu.number_label)
                  ELSE slugify(lu.unit_type::text)
                END AS token
              FROM legal_units lu
            ),
            doc_root AS (
              SELECT d.id AS document_id, slugify(d.doc_id)::text AS root_token
              FROM legal_documents d
            ),
            orphans AS (
              SELECT lu.id, (dr.root_token || '.' || t.token) AS full_path
              FROM legal_units lu
              JOIN tokens t ON t.id = lu.id
              JOIN doc_root dr ON dr.document_id = lu.document_id
              WHERE lu.unit_path IS NULL
            )
            UPDATE legal_units lu
            SET unit_path = text2ltree(o.full_path)
            FROM orphans o
            WHERE lu.id = o.id
              AND lu.unit_path IS NULL;
            """
        )
    )

    # Optional: validate with simple sanity checks (advisory warnings via NOTICE)
    op.execute(
        sa.text(
            """
            DO $$
            DECLARE missing int;
            BEGIN
              SELECT COUNT(*) INTO missing FROM legal_units WHERE unit_path IS NULL;
              IF missing > 0 THEN
                RAISE NOTICE 'unit_path backfill: % rows still NULL', missing;
              END IF;
            END $$;
            """
        )
    )


def downgrade() -> None:
    # Non-destructive: keep computed unit_path. No-op on downgrade.
    return
