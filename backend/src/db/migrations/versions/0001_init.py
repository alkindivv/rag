from __future__ import annotations
"""Initial database schema with SQLite fallback."""

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create tables and indexes."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        op.create_table(
            "legal_documents",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("form", sa.String(50), nullable=False),
            sa.Column("number", sa.String(50), nullable=False),
            sa.Column("year", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(50), nullable=False),
            sa.Column("title", sa.Text()),
        )
        op.create_table(
            "legal_units",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("document_id", sa.Integer, index=True),
            sa.Column("unit_id", sa.String(500), nullable=False),
            sa.Column("unit_type", sa.String(50), nullable=False),
            sa.Column("parent_unit_id", sa.String(500)),
            sa.Column("ordinal", sa.String(50)),
            sa.Column("bm25_body", sa.Text()),
            sa.Column("citation", sa.Text()),
        )
        op.create_table(
            "document_vectors",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("document_id", sa.Integer, index=True),
            sa.Column("unit_id", sa.String(500), nullable=False),
            sa.Column("embedding", sa.LargeBinary(), nullable=False),
        )
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "legal_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("form", sa.Enum("uu", "pp", name="docform"), nullable=False),
        sa.Column("number", sa.String(50), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("status", sa.Enum("berlaku", "tidak_berlaku", name="docstatus"), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("form", "number", "year", name="uq_doc_identity"),
    )
    op.create_table(
        "legal_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True),
        sa.Column("unit_id", sa.String(500), nullable=False),
        sa.Column("unit_type", sa.Enum("buku", "bab", "bagian", "paragraf", "pasal", "ayat", "angka", "huruf", name="unittype"), nullable=False),
        sa.Column("parent_unit_id", sa.String(500)),
        sa.Column("ordinal", sa.String(50)),
        sa.Column("ordinal_int", sa.Integer()),
        sa.Column("title", sa.Text()),
        sa.Column("bm25_body", sa.Text()),
        sa.Column("content_vector", postgresql.TSVECTOR, server_default=sa.text("to_tsvector('indonesian', coalesce(bm25_body,''))"), nullable=False),
        sa.Column("path", sa.Text()),
        sa.Column("citation", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("idx_units_doc_unitid", "legal_units", ["document_id", "unit_id"], unique=True)
    op.create_index("idx_units_type_ord", "legal_units", ["unit_type", "ordinal_int"])
    op.create_index("idx_units_bm25_fts", "legal_units", ["content_vector"], postgresql_using="gin")
    op.create_table(
        "document_vectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True),
        sa.Column("unit_id", sa.String(500), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=False),
        sa.Column("embedding_model", sa.String(100)),
        sa.Column("doc_form", sa.Enum("uu", "pp", name="docform"), nullable=False),
        sa.Column("doc_year", sa.Integer(), nullable=False),
        sa.Column("doc_number", sa.String(100), nullable=False),
        sa.Column("doc_status", sa.Enum("berlaku", "tidak_berlaku", name="docstatus"), nullable=False),
        sa.Column("pasal_number", sa.String(20)),
        sa.Column("hierarchy_path", sa.Text()),
        sa.Column("token_count", sa.Integer(), server_default="0"),
        sa.Column("char_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index(
        "idx_vec_embedding_hnsw",
        "document_vectors",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("idx_vec_doc_meta", "document_vectors", ["doc_form", "doc_year", "doc_number"])


def downgrade() -> None:
    """Drop created tables."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        op.drop_table("document_vectors")
        op.drop_table("legal_units")
        op.drop_table("legal_documents")
        return

    op.drop_index("idx_vec_doc_meta", table_name="document_vectors")
    op.drop_index("idx_vec_embedding_hnsw", table_name="document_vectors")
    op.drop_table("document_vectors")
    op.drop_index("idx_units_bm25_fts", table_name="legal_units")
    op.drop_index("idx_units_type_ord", table_name="legal_units")
    op.drop_index("idx_units_doc_unitid", table_name="legal_units")
    op.drop_table("legal_units")
    op.drop_table("legal_documents")
