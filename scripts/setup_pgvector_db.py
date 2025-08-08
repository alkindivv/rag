#!/usr/bin/env python3
"""
Production PostgreSQL + pgvector Database Setup
Clean setup for legal document storage and semantic search.

Sets up:
- PostgreSQL with pgvector extension
- Legal document storage with full metadata
- Vector storage optimized for semantic search
- Proper indexes for production performance
- tsvector preparation for future FTS

Author: Production System
Purpose: One-script setup for complete legal document system
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.document_storage import LegalDocument, Base as DocumentBase
    from models.vector_storage import DocumentVector, VectorSearchLog, Base as VectorBase
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the refactor directory and src/ exists")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDatabaseSetup:
    """
    Production-ready database setup for legal document system.

    Features:
    - Clean table creation
    - Optimized indexes
    - pgvector extension
    - Full-text search preparation
    - Performance monitoring setup
    """

    def __init__(self):
        """Initialize with environment configuration."""
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "postgres")
        self.username = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "")

        self.connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def create_database_if_needed(self):
        """Create database if it doesn't exist."""
        try:
            # Connect to postgres database to create target database
            postgres_conn_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/postgres"

            logger.info(f"🔗 Connecting to PostgreSQL server at {self.host}:{self.port}")

            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if target database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.database,)
            )

            if cursor.fetchone():
                logger.info(f"✅ Database '{self.database}' already exists")
            else:
                # Create database
                cursor.execute(f'CREATE DATABASE "{self.database}"')
                logger.info(f"✅ Created database '{self.database}'")

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Error with database creation: {e}")
            raise

    def setup_extensions(self):
        """Install required PostgreSQL extensions."""
        try:
            engine = create_engine(self.connection_string)

            with engine.connect() as conn:
                logger.info("🔧 Installing PostgreSQL extensions...")

                # pgvector for vector similarity search
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("✅ pgvector extension installed")

                # pg_trgm for trigram text search
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                logger.info("✅ pg_trgm extension installed")

                # uuid-ossp for UUID generation functions
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                logger.info("✅ uuid-ossp extension installed")

                conn.commit()

                # Verify extensions
                result = conn.execute(text("""
                    SELECT extname FROM pg_extension
                    WHERE extname IN ('vector', 'pg_trgm', 'uuid-ossp')
                    ORDER BY extname
                """)).fetchall()

                installed = [row[0] for row in result]
                logger.info(f"✅ Extensions verified: {', '.join(installed)}")

        except Exception as e:
            logger.error(f"❌ Error installing extensions: {e}")
            raise

    def create_tables(self):
        """Create all database tables with proper structure."""
        try:
            engine = create_engine(self.connection_string)

            logger.info("📝 Creating database tables...")

            # Create document storage tables
            DocumentBase.metadata.create_all(bind=engine)
            logger.info("✅ Legal document tables created")

            # Create vector storage tables
            VectorBase.metadata.create_all(bind=engine)
            logger.info("✅ Vector storage tables created")

            # Verify tables were created
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('legal_documents', 'document_vectors', 'vector_search_logs')
                    ORDER BY table_name
                """)).fetchall()

                tables = [row[0] for row in result]
                logger.info(f"✅ Tables created: {', '.join(tables)}")

        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise

    def create_production_indexes(self):
        """Create additional production-optimized indexes."""
        try:
            engine = create_engine(self.connection_string)

            with engine.connect() as conn:
                logger.info("⚡ Creating production performance indexes...")

                # Additional indexes for legal_documents
                production_indexes = [
                    # Text search preparation for FTS
                    """
                    CREATE INDEX IF NOT EXISTS idx_legal_doc_title_fts
                    ON legal_documents USING GIN (to_tsvector('indonesian', title))
                    """,

                    # Content search for large documents
                    """
                    CREATE INDEX IF NOT EXISTS idx_legal_doc_content_fts
                    ON legal_documents USING GIN (to_tsvector('indonesian', content))
                    """,

                    # Combined search across title and content
                    """
                    CREATE INDEX IF NOT EXISTS idx_legal_doc_combined_fts
                    ON legal_documents USING GIN (
                        to_tsvector('indonesian', title || ' ' || COALESCE(content, ''))
                    )
                    """,

                    # Processing status for pipeline monitoring
                    """
                    CREATE INDEX IF NOT EXISTS idx_legal_doc_processing_status
                    ON legal_documents (processing_status, created_at)
                    """,

                    # Document relationships for legal research
                    """
                    CREATE INDEX IF NOT EXISTS idx_legal_doc_relationships
                    ON legal_documents USING GIN (amends || revokes || amended_by || revoked_by)
                    """,

                    # Vector search optimization indexes
                    """
                    CREATE INDEX IF NOT EXISTS idx_doc_vector_similarity_search
                    ON document_vectors (doc_type, doc_status, doc_year)
                    """,

                    # Legal hierarchy search
                    """
                    CREATE INDEX IF NOT EXISTS idx_doc_vector_legal_structure
                    ON document_vectors (doc_type, bab_number, pasal_number, ayat_number)
                    """,

                    # Content analysis indexes
                    """
                    CREATE INDEX IF NOT EXISTS idx_doc_vector_content_analysis
                    ON document_vectors (content_type, token_count, char_count)
                    """,

                    # Search logging indexes for analytics
                    """
                    CREATE INDEX IF NOT EXISTS idx_search_log_performance
                    ON vector_search_logs (searched_at, search_duration_ms)
                    """,

                    """
                    CREATE INDEX IF NOT EXISTS idx_search_log_patterns
                    ON vector_search_logs (query_vector_hash, results_found)
                    """
                ]

                for i, index_sql in enumerate(production_indexes, 1):
                    try:
                        conn.execute(text(index_sql))
                        index_name = index_sql.split('idx_')[1].split()[0] if 'idx_' in index_sql else f'index_{i}'
                        logger.info(f"✅ Created index: {index_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Index creation warning: {e}")

                conn.commit()
                logger.info("✅ Production indexes created successfully")

        except Exception as e:
            logger.error(f"❌ Error creating production indexes: {e}")
            raise

    def setup_tsvector_triggers(self):
        """Setup tsvector triggers for automatic full-text search updates."""
        try:
            engine = create_engine(self.connection_string)

            with engine.connect() as conn:
                logger.info("🔄 Setting up tsvector triggers for FTS...")

                # Function to update content_vector automatically
                trigger_function = """
                CREATE OR REPLACE FUNCTION update_legal_document_tsvector()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.content_vector := to_tsvector('indonesian',
                        COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, '')
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """

                conn.execute(text(trigger_function))
                logger.info("✅ Created tsvector update function")

                # Trigger to automatically update tsvector on insert/update
                trigger_sql = """
                DROP TRIGGER IF EXISTS trigger_update_legal_document_tsvector ON legal_documents;

                CREATE TRIGGER trigger_update_legal_document_tsvector
                    BEFORE INSERT OR UPDATE OF title, content
                    ON legal_documents
                    FOR EACH ROW
                    EXECUTE FUNCTION update_legal_document_tsvector();
                """

                conn.execute(text(trigger_sql))
                logger.info("✅ Created tsvector trigger")

                # Update existing records
                update_existing = """
                UPDATE legal_documents
                SET content_vector = to_tsvector('indonesian',
                    COALESCE(title, '') || ' ' || COALESCE(content, '')
                )
                WHERE content_vector IS NULL;
                """

                result = conn.execute(text(update_existing))
                logger.info(f"✅ Updated tsvector for existing records")

                conn.commit()

        except Exception as e:
            logger.error(f"❌ Error setting up tsvector triggers: {e}")
            raise

    def create_uuid5_functions(self):
        """Create UUID5 helper functions for consistency."""
        try:
            engine = create_engine(self.connection_string)

            with engine.connect() as conn:
                logger.info("🔑 Creating UUID5 helper functions...")

                # Function to generate consistent document UUIDs
                uuid5_function = """
                CREATE OR REPLACE FUNCTION generate_document_uuid5(
                    doc_form TEXT,
                    doc_number TEXT,
                    doc_year INTEGER
                ) RETURNS UUID AS $$
                DECLARE
                    namespace_uuid UUID := '6ba7b810-9dad-11d1-80b4-00c04fd430c8'::UUID;
                    doc_key TEXT;
                BEGIN
                    doc_key := doc_form || '-' || doc_number || '-' || doc_year;
                    RETURN uuid_generate_v5(namespace_uuid, doc_key);
                END;
                $$ LANGUAGE plpgsql;
                """

                conn.execute(text(uuid5_function))
                logger.info("✅ Created document UUID5 function")

                # Function to generate consistent vector UUIDs
                vector_uuid5_function = """
                CREATE OR REPLACE FUNCTION generate_vector_uuid5(
                    doc_id UUID,
                    content_hash TEXT
                ) RETURNS UUID AS $$
                DECLARE
                    namespace_uuid UUID := '6ba7b811-9dad-11d1-80b4-00c04fd430c8'::UUID;
                    vector_key TEXT;
                BEGIN
                    vector_key := doc_id::TEXT || '-' || content_hash;
                    RETURN uuid_generate_v5(namespace_uuid, vector_key);
                END;
                $$ LANGUAGE plpgsql;
                """

                conn.execute(text(vector_uuid5_function))
                logger.info("✅ Created vector UUID5 function")

                conn.commit()

        except Exception as e:
            logger.error(f"❌ Error creating UUID5 functions: {e}")
            raise

    def verify_setup(self):
        """Verify the database setup is working correctly."""
        try:
            engine = create_engine(self.connection_string)

            with engine.connect() as conn:
                logger.info("🔍 Verifying database setup...")

                # Test pgvector functionality
                result = conn.execute(text("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance")).fetchone()
                if result:
                    logger.info(f"✅ pgvector working (test distance: {result[0]:.3f})")

                # Test tsvector functionality
                result = conn.execute(text("""
                    SELECT to_tsvector('indonesian', 'Undang-undang tentang hak asasi manusia') @@
                           plainto_tsquery('indonesian', 'hak asasi') as matches
                """)).fetchone()
                if result and result[0]:
                    logger.info("✅ tsvector FTS working")

                # Test UUID5 functions
                result = conn.execute(text("""
                    SELECT generate_document_uuid5('UU', '39', 1999) as uuid
                """)).fetchone()
                if result:
                    logger.info(f"✅ UUID5 function working (sample: {result[0]})")

                # Count existing data
                tables_info = {}
                for table in ['legal_documents', 'document_vectors', 'vector_search_logs']:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                        tables_info[table] = result[0] if result else 0
                    except:
                        tables_info[table] = 0

                logger.info("📊 Current data counts:")
                for table, count in tables_info.items():
                    logger.info(f"   {table}: {count} records")

                # Check indexes
                result = conn.execute(text("""
                    SELECT COUNT(*) as index_count
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND (indexname LIKE 'idx_%' OR indexname LIKE '%_pkey' OR indexname LIKE '%vector%')
                """)).fetchone()

                index_count = result[0] if result else 0
                logger.info(f"✅ Database indexes: {index_count} total")

        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            raise

    def setup_all(self):
        """Run complete production database setup."""
        try:
            logger.info("🚀 Starting production database setup...")
            logger.info("=" * 60)
            logger.info(f"📍 Target: {self.host}:{self.port}/{self.database}")
            logger.info("=" * 60)

            # Step 1: Create database if needed
            self.create_database_if_needed()

            # Step 2: Install extensions
            self.setup_extensions()

            # Step 3: Create tables
            self.create_tables()

            # Step 4: Create production indexes
            self.create_production_indexes()

            # Step 5: Setup tsvector for FTS
            self.setup_tsvector_triggers()

            # Step 6: Create UUID5 helper functions
            self.create_uuid5_functions()

            # Step 7: Verify everything is working
            self.verify_setup()

            logger.info("=" * 60)
            logger.info("🎉 Production database setup completed successfully!")
            logger.info(f"📍 Database: {self.host}:{self.port}/{self.database}")
            logger.info("✅ Ready for legal document storage and semantic search")
            logger.info("💡 Features enabled:")
            logger.info("   - Complete legal document metadata storage")
            logger.info("   - Optimized vector similarity search")
            logger.info("   - Full-text search preparation")
            logger.info("   - UUID5 consistency functions")
            logger.info("   - Production performance indexes")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise


def main():
    """Main setup function."""
    try:
        logger.info("Production PostgreSQL + pgvector Database Setup")
        logger.info("Legal Document Storage & Semantic Search")
        logger.info("=" * 60)

        # Initialize setup
        setup = ProductionDatabaseSetup()

        logger.info(f"Host: {setup.host}")
        logger.info(f"Port: {setup.port}")
        logger.info(f"Database: {setup.database}")
        logger.info(f"User: {setup.username}")
        logger.info("=" * 60)

        # Confirm setup
        if os.getenv("AUTO_CONFIRM") != "true":
            response = input("Proceed with production database setup? (y/N): ")
            if response.lower() != 'y':
                logger.info("Setup cancelled by user")
                return

        # Run setup
        setup.setup_all()

    except KeyboardInterrupt:
        logger.info("Setup cancelled by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
