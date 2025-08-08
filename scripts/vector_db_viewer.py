"""
Vector Database Viewer
ONE FILE to visualize your actual chunking and embedding results.

KISS Implementation: Simple but powerful visualization of vector DB content.
No overengineering, just direct visualization of what's in your database.

Author: KISS Principle Implementation
Purpose: See exactly what your vector database looks like after processing
"""

import json
import os
import sys
import re
import time
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import pattern manager
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from utils.pattern_manager import PatternManager
    PATTERN_MANAGER_AVAILABLE = True
except ImportError:
    PATTERN_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️  google.generativeai not available - embeddings will be skipped")


@dataclass
class Chunk:
    """Simple chunk representation."""
    content: str
    citation: str
    keywords: List[str]
    tokens: int = 0

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = len(self.content.split())


@dataclass
class VectorRecord:
    """Simple vector database record representation."""
    id: str
    content: str
    citation: str
    keywords: List[str]
    embedding: List[float]
    token_count: int
    content_length: int


class SimpleChunker:
    """Inline simple chunker - no external dependencies."""

    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

        # Use pattern manager if available, otherwise fallback to simple patterns
        if PATTERN_MANAGER_AVAILABLE:
            self.pattern_manager = PatternManager()
            self.use_pattern_manager = True
        else:
            self.use_pattern_manager = False
            # Fallback pattern - only match Pasal headers, not references
            self.pasal_pattern = re.compile(
                r'(?:^|\d+\.\s+)Pasal\s+(\d+(?:[A-Z])?)',
                re.IGNORECASE | re.MULTILINE
            )
            self.bab_pattern = re.compile(
                r'\bBAB\s+([IVX]+|\d+)\s*[:\-]?\s*([A-Z\s]+?)(?=\n|$|BAB|\bPasal)',
                re.IGNORECASE | re.MULTILINE
            )

        self.legal_keywords = [
            'pasal', 'ayat', 'huruf', 'bab', 'undang-undang', 'peraturan',
            'pemerintah', 'negara', 'republik', 'indonesia', 'presiden', 'menteri'
        ]

    def chunk_document(self, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk document into pasal-based chunks."""
        if not text or not text.strip():
            return []

        chunks = []

        if self.use_pattern_manager:
            # Use pattern_manager for accurate pasal detection
            articles = self.pattern_manager.find_articles(text)
            if not articles:
                # No pasal found - create single chunk
                chunk = Chunk(
                    content=text,
                    citation=f"{doc_metadata.get('type', 'Document')} {doc_metadata.get('title', 'Unknown')}",
                    keywords=self._extract_keywords(text)
                )
                return [chunk]

            # Find BAB for context
            chapters = self.pattern_manager.find_chapters(text)
            bab_context = f"BAB {chapters[0].number}" if chapters else ""

            # Find pasal boundaries - only match Pasal headers, not references
            pasal_pattern = re.compile(
                r'(?:^|\d+\.\s+)Pasal\s+(\d+(?:[A-Z])?)',
                re.IGNORECASE | re.MULTILINE
            )
            pasal_matches = list(pasal_pattern.finditer(text))
        else:
            # Fallback to simple pattern matching
            pasal_matches = list(self.pasal_pattern.finditer(text))
            if not pasal_matches:
                # No pasal found - create single chunk
                chunk = Chunk(
                    content=text,
                    citation=f"{doc_metadata.get('type', 'Document')} {doc_metadata.get('title', 'Unknown')}",
                    keywords=self._extract_keywords(text)
                )
                return [chunk]

            # Find BAB for context
            bab_match = self.bab_pattern.search(text)
            bab_context = f"BAB {bab_match.group(1)}" if bab_match else ""

        # Process each pasal
        for i, match in enumerate(pasal_matches):
            pasal_num = match.group(1)
            start = match.start()
            end = pasal_matches[i + 1].start() if i + 1 < len(pasal_matches) else len(text)

            pasal_content = text[start:end].strip()
            if not pasal_content:
                continue

            citation = self._build_citation(bab_context, pasal_num, doc_metadata)
            keywords = self._extract_keywords(pasal_content)

            # Always keep complete pasal content - text is already cleaned
            chunk = Chunk(
                content=pasal_content,
                citation=citation,
                keywords=keywords
            )
            chunks.append(chunk)

        return chunks

    def _build_citation(self, bab_context: str, pasal_num: str, doc_metadata: Dict[str, Any]) -> str:
        """Build citation."""
        parts = []
        if bab_context:
            parts.append(bab_context)
        parts.append(f"Pasal {pasal_num}")

        doc_type = doc_metadata.get('type', '').upper()
        number = doc_metadata.get('number', '')
        year = doc_metadata.get('year', '')

        if doc_type and number and year:
            if doc_type == 'UNDANG-UNDANG':
                doc_ref = f"UU No. {number} Tahun {year}"
            else:
                doc_ref = f"{doc_type} No. {number} Tahun {year}"
            parts.append(doc_ref)

        return " ".join(parts)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords."""
        if not content:
            return []

        content_lower = content.lower()
        found_keywords = []

        # Legal terms
        for keyword in self.legal_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)

        # Important terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content_lower)
        word_freq = {}
        stop_words = {'yang', 'dengan', 'untuk', 'dari', 'dalam', 'pada', 'adalah', 'akan'}

        for word in words:
            if word not in stop_words and len(word) >= 4:
                word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        found_keywords.extend([word for word, freq in top_words if word not in found_keywords])

        return found_keywords[:10]

    def _split_large_pasal(self, content: str, base_citation: str, keywords: List[str]) -> List[Chunk]:
        """Split large pasal by ayat or sentences."""
        chunks = []
        ayat_matches = list(self.ayat_pattern.finditer(content))

        if len(ayat_matches) > 1:
            # Split by ayat
            for i, match in enumerate(ayat_matches):
                ayat_num = match.group(1)
                start = match.start()
                end = ayat_matches[i + 1].start() if i + 1 < len(ayat_matches) else len(content)

                ayat_content = content[start:end].strip()
                if ayat_content and len(ayat_content) > 10:
                    ayat_citation = f"{base_citation} ayat ({ayat_num})"
                    chunks.append(Chunk(
                        content=ayat_content,
                        citation=ayat_citation,
                        keywords=keywords
                    ))
        else:
            # Force split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current_chunk = ""
            current_tokens = 0

            for sentence in sentences:
                if not sentence.strip():
                    continue

                sentence_tokens = len(sentence.split())

                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    chunk_content = current_chunk.strip()
                    if len(chunk_content) > 10:
                        chunks.append(Chunk(
                            content=chunk_content,
                            citation=base_citation,
                            keywords=keywords
                        ))
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_tokens += sentence_tokens

            if current_chunk and len(current_chunk.strip()) > 10:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    citation=base_citation,
                    keywords=keywords
                ))

        return chunks if chunks else [Chunk(content=content, citation=base_citation, keywords=keywords)]


class SimpleEmbedder:
    """Inline simple embedder - no external dependencies."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key and GENAI_AVAILABLE:
            genai.configure(api_key=api_key)

    def embed_chunks(self, chunks: List[Chunk], doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks."""
        if not self.api_key or not GENAI_AVAILABLE:
            # Return chunks without embeddings
            return [self._chunk_to_dict(chunk, [], doc_metadata) for chunk in chunks]

        results = []
        for chunk in chunks:
            try:
                # Prepare content for embedding
                embedding_text = self._prepare_embedding_text(chunk, doc_metadata)

                # Get embedding
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=embedding_text,
                    task_type="retrieval_document"
                )

                embedding = result.get('embedding', [])
                results.append(self._chunk_to_dict(chunk, embedding, doc_metadata))

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Embedding failed for chunk: {e}")
                results.append(self._chunk_to_dict(chunk, [], doc_metadata))

        return results

    def _prepare_embedding_text(self, chunk: Chunk, doc_metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding."""
        context_parts = [
            f"Dokumen: {doc_metadata.get('title', 'Unknown')}",
            f"Jenis: {doc_metadata.get('type', 'Unknown')}",
            f"Sitasi: {chunk.citation}",
            f"Kata Kunci: {', '.join(chunk.keywords[:5])}"
        ]

        context_prefix = " | ".join(context_parts)
        return f"{context_prefix}\n\n{chunk.content}"

    def _chunk_to_dict(self, chunk: Chunk, embedding: List[float], doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            'content': chunk.content,
            'citation_path': chunk.citation,
            'semantic_keywords': chunk.keywords,
            'token_count': chunk.tokens,
            'embedding': embedding,
            'content_length': len(chunk.content),
            'has_embedding': len(embedding) > 0
        }

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query."""
        if not self.api_key or not GENAI_AVAILABLE:
            return []

        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result.get('embedding', [])
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []


class VectorDBViewer:
    """
    KISS Vector Database Viewer

    Shows you exactly what your vector database looks like after chunking and embedding.
    No complex visualization libraries, just clear text output.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with minimal setup."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.records: List[VectorRecord] = []
        self.chunker = SimpleChunker()
        self.embedder = SimpleEmbedder(self.api_key)

    def process_document(self, text_file_path: str, document_metadata: Dict[str, Any]) -> None:
        """Process document and show vector DB representation."""
        print("🔄 Processing document for vector database...")

        # Read document
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return

        print(f"📄 Document loaded: {len(text)} characters")

        # Step 1: Chunk the document
        print("✂️  Chunking document...")
        start_time = time.time()
        chunks = self.chunker.chunk_document(text, document_metadata)
        chunk_time = time.time() - start_time
        print(f"   Generated {len(chunks)} chunks in {chunk_time:.2f}s")

        # Step 2: Create embeddings
        print("🧠 Creating embeddings...")
        start_time = time.time()
        embedded_chunks = self.embedder.embed_chunks(chunks, document_metadata)
        embed_time = time.time() - start_time

        if self.api_key and GENAI_AVAILABLE:
            print(f"   Created embeddings in {embed_time:.2f}s")
        else:
            print("   No embeddings created (no API key or library)")

        # Convert to vector records
        for i, chunk_data in enumerate(embedded_chunks):
            record = VectorRecord(
                id=f"doc_{i+1:03d}",
                content=chunk_data['content'],
                citation=chunk_data['citation_path'],
                keywords=chunk_data['semantic_keywords'],
                embedding=chunk_data['embedding'],
                token_count=chunk_data['token_count'],
                content_length=chunk_data['content_length']
            )
            self.records.append(record)

    def show_database_overview(self) -> None:
        """Show high-level overview of vector database."""
        print("\n" + "="*80)
        print("📊 VECTOR DATABASE OVERVIEW")
        print("="*80)

        if not self.records:
            print("❌ No records in database")
            return

        total_records = len(self.records)
        total_content_chars = sum(r.content_length for r in self.records)
        total_tokens = sum(r.token_count for r in self.records)
        has_embeddings = any(len(r.embedding) > 0 for r in self.records)
        embedding_dim = len(self.records[0].embedding) if has_embeddings else 0
        unique_citations = len(set(r.citation for r in self.records))

        print(f"📈 Total Records: {total_records}")
        print(f"📝 Total Content: {total_content_chars:,} characters")
        print(f"🔤 Total Tokens: {total_tokens:,}")
        print(f"📋 Unique Citations: {unique_citations}")
        print(f"🧠 Embeddings: {'✅ Present' if has_embeddings else '❌ Missing'}")
        if has_embeddings:
            print(f"📐 Embedding Dimension: {embedding_dim}")

        print(f"\n📊 Record Statistics:")
        content_lengths = [r.content_length for r in self.records]
        token_counts = [r.token_count for r in self.records]

        print(f"   Content Length - Min: {min(content_lengths)}, Max: {max(content_lengths)}, Avg: {sum(content_lengths)/len(content_lengths):.0f}")
        print(f"   Token Count - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.0f}")

    def show_detailed_records(self, limit: int = 20) -> None:
        """Show detailed view of vector records."""
        print(f"\n📋 DETAILED RECORD VIEW (showing first {limit} records)")
        print("="*80)

        for i, record in enumerate(self.records[:limit]):
            print(f"\n🔖 Record ID: {record.id}")
            print(f"📍 Citation: {record.citation}")
            print(f"🏷️  Keywords: {', '.join(record.keywords[:5])}")
            print(f"📊 Stats: {record.content_length} chars, {record.token_count} tokens")
            if record.embedding:
                print(f"🧠 Embedding: [{record.embedding[0]:.4f}, {record.embedding[1]:.4f}, ..., {record.embedding[-1]:.4f}] (dim: {len(record.embedding)})")
            else:
                print(f"🧠 Embedding: None")
            print(f"📄 Content:")
            # Show actual content with proper formatting
            content_lines = record.content.split('\n')
            for line in content_lines[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"   {line.strip()}")
            if len(content_lines) > 5:
                print(f"   ... ({len(content_lines) - 5} more lines)")

            if i < len(self.records[:limit]) - 1:
                print("-" * 80)

    def show_content_analysis(self) -> None:
        """Analyze content distribution and quality."""
        print(f"\n🔍 CONTENT ANALYSIS")
        print("="*80)

        if not self.records:
            return

        # Citation analysis
        citation_counts = {}
        for record in self.records:
            base_citation = record.citation.split(' ayat')[0].split(' huruf')[0]
            citation_counts[base_citation] = citation_counts.get(base_citation, 0) + 1

        print(f"📋 Citation Distribution:")
        for citation, count in sorted(citation_counts.items())[:10]:
            print(f"   {citation}: {count} chunks")
        if len(citation_counts) > 10:
            print(f"   ... and {len(citation_counts) - 10} more citations")

        # Keyword analysis
        all_keywords = []
        for record in self.records:
            all_keywords.extend(record.keywords)

        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1

        print(f"\n🏷️  Top Keywords:")
        for keyword, freq in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"   {keyword}: {freq} occurrences")

    def export_database_json(self, filename: str = "vector_database_export.json") -> None:
        """Export vector database to JSON for inspection."""
        print(f"\n💾 Exporting database to {filename}...")

        export_data = {
            "metadata": {
                "total_records": len(self.records),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "has_embeddings": any(len(r.embedding) > 0 for r in self.records),
                "embedding_dimension": len(self.records[0].embedding) if self.records and self.records[0].embedding else 0
            },
            "records": []
        }

        for record in self.records:
            export_record = {
                "id": record.id,
                "citation": record.citation,
                "keywords": record.keywords,
                "token_count": record.token_count,
                "content_length": record.content_length,
                "content": record.content,
                "has_embedding": len(record.embedding) > 0,
                "embedding_preview": record.embedding[:5] if record.embedding else [],
                "embedding_full": record.embedding if record.embedding else []
            }
            export_data["records"].append(export_record)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Database exported successfully")
            print(f"📊 Contains {len(self.records)} records with full content and embeddings")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def search_similar_content(self, query: str, top_k: int = 3) -> None:
        """Simple similarity search in the vector database."""
        if not self.records or not any(r.embedding for r in self.records):
            print("⚠️  Cannot perform similarity search - no embeddings available")
            return

        print(f"\n🔍 SIMILARITY SEARCH: '{query}'")
        print("="*80)

        # Get query embedding
        query_embedding = self.embedder.embed_query(query)
        if not query_embedding:
            print("❌ Query embedding failed")
            return

        # Calculate similarities
        similarities = []
        for record in self.records:
            if record.embedding:
                similarity = self._cosine_similarity(query_embedding, record.embedding)
                similarities.append((similarity, record))

        # Sort and show top results
        similarities.sort(key=lambda x: x[0], reverse=True)

        print(f"📊 Found {len(similarities)} records with embeddings")
        print(f"🎯 Top {top_k} most similar:")

        for i, (similarity, record) in enumerate(similarities[:top_k]):
            print(f"\n#{i+1} - Score: {similarity:.4f}")
            print(f"📍 {record.citation}")
            print(f"🏷️  {', '.join(record.keywords[:3])}")
            content_preview = record.content.replace('\n', ' ').strip()[:150]
            print(f"📄 {content_preview}...")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


def main():
    """Main function to run vector database viewer."""
    print("🗂️  VECTOR DATABASE VIEWER")
    print("KISS Implementation: Direct visualization of your chunking and embedding results")
    print("="*80)

    # Check for text file
    if len(sys.argv) < 2:
        print("Usage: python vector_db_viewer.py <text_file_path>")
        print("Example: python vector_db_viewer.py document.txt")
        return

    text_file = sys.argv[1]
    if not os.path.exists(text_file):
        print(f"❌ File not found: {text_file}")
        return

    # Document metadata (adjust as needed)
    doc_metadata = {
        'title': os.path.basename(text_file),
        'type': 'undang-undang',
        'number': '3',
        'year': '2025'
    }

    # Initialize viewer
    viewer = VectorDBViewer()

    # Process document
    viewer.process_document(text_file, doc_metadata)

    # Show database views
    viewer.show_database_overview()
    viewer.show_detailed_records(limit=20)
    viewer.show_content_analysis()

    # Export to JSON
    viewer.export_database_json()

    # Interactive search (if embeddings available)
    if viewer.api_key and GENAI_AVAILABLE and any(len(r.embedding) > 0 for r in viewer.records):
        print(f"\n🔍 Try similarity search:")
        while True:
            try:
                query = input("Enter search query (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    viewer.search_similar_content(query, top_k=3)
            except KeyboardInterrupt:
                break

    print(f"\n✨ Vector database visualization complete!")
    print(f"📄 Check vector_database_export.json for full database export")


if __name__ == "__main__":
    main()
