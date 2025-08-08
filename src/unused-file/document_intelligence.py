"""
Document Intelligence Service
Auto-extracts legal concepts, relationships, and citation networks from documents
"""

import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LegalConcept:
    """Extracted legal concept with metadata."""
    term: str
    category: str
    definition: str
    frequency: int
    importance_score: float
    context_examples: List[str]
    related_terms: List[str]


@dataclass
class DocumentRelationship:
    """Relationship between documents or provisions."""
    source_document: str
    target_document: str
    relationship_type: str
    strength: float
    evidence: List[str]
    description: str


@dataclass
class CitationNetwork:
    """Citation network analysis result."""
    total_citations: int
    internal_citations: int
    external_citations: int
    most_cited_provisions: List[Dict[str, Any]]
    citation_clusters: List[Dict[str, Any]]
    authority_score: float


@dataclass
class DocumentIntelligence:
    """Complete document intelligence analysis result."""
    document_id: str
    document_title: str
    legal_concepts: List[LegalConcept]
    document_relationships: List[DocumentRelationship]
    citation_network: CitationNetwork
    legal_structure: Dict[str, Any]
    key_provisions: List[Dict[str, Any]]
    complexity_score: float
    analysis_timestamp: str


class DocumentIntelligenceService:
    """
    Advanced document intelligence service for legal documents.
    Automatically extracts legal concepts, analyzes relationships,
    and builds citation networks.
    """

    def __init__(self):
        self.legal_patterns = self._load_legal_patterns()
        self.concept_categories = self._load_concept_categories()
        self.citation_patterns = self._load_citation_patterns()

    def _load_legal_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying legal concepts."""
        return {
            "definitions": [
                r"(?:yang dimaksud dengan|pengertian|definisi)\s+([^,\.]+)",
                r"([A-Z][a-z\s]+),?\s+(?:yang selanjutnya disebut|selanjutnya disebut)\s+([A-Z][a-z]+)",
                r"dalam (?:undang-undang|peraturan) ini\s+([^,\.]+)\s+adalah\s+([^\.]+)"
            ],
            "obligations": [
                r"wajib\s+([^,\.]+)",
                r"berkewajiban\s+([^,\.]+)",
                r"harus\s+([^,\.]+)",
                r"diwajibkan\s+([^,\.]+)"
            ],
            "rights": [
                r"berhak\s+([^,\.]+)",
                r"hak\s+untuk\s+([^,\.]+)",
                r"dapat\s+([^,\.]+)",
                r"diperbolehkan\s+([^,\.]+)"
            ],
            "prohibitions": [
                r"dilarang\s+([^,\.]+)",
                r"tidak boleh\s+([^,\.]+)",
                r"tidak dapat\s+([^,\.]+)",
                r"diperlarang\s+([^,\.]+)"
            ],
            "sanctions": [
                r"dikenai sanksi\s+([^,\.]+)",
                r"dipidana\s+([^,\.]+)",
                r"denda\s+([^,\.]+)",
                r"hukuman\s+([^,\.]+)"
            ],
            "procedures": [
                r"prosedur\s+([^,\.]+)",
                r"tata cara\s+([^,\.]+)",
                r"mekanisme\s+([^,\.]+)",
                r"langkah-langkah\s+([^,\.]+)"
            ]
        }

    def _load_concept_categories(self) -> Dict[str, List[str]]:
        """Load categories for legal concept classification."""
        return {
            "entities": [
                "perseroan", "korporasi", "badan hukum", "yayasan", "koperasi",
                "perusahaan", "firma", "cv", "pt", "tbk"
            ],
            "roles": [
                "direksi", "komisaris", "pemegang saham", "direktur", "manajer",
                "pegawai", "karyawan", "pekerja", "buruh", "pengurus"
            ],
            "documents": [
                "akta", "surat", "sertifikat", "izin", "lisensi", "kontrak",
                "perjanjian", "memorandum", "laporan", "pernyataan"
            ],
            "processes": [
                "pendirian", "pembubaran", "merger", "akuisisi", "likuidasi",
                "reorganisasi", "transformasi", "konsolidasi"
            ],
            "financial": [
                "modal", "saham", "dividen", "laba", "rugi", "aset", "utang",
                "investasi", "pendapatan", "biaya", "pajak"
            ],
            "governance": [
                "rups", "rapat", "keputusan", "voting", "suara", "persetujuan",
                "pengesahan", "ratifikasi", "validasi"
            ],
            "compliance": [
                "pelaporan", "audit", "pemeriksaan", "evaluasi", "monitoring",
                "supervisi", "pengawasan", "kontrol"
            ]
        }

    def _load_citation_patterns(self) -> List[str]:
        """Load patterns for identifying legal citations."""
        return [
            r"[Pp]asal\s+(\d+)(?:\s+ayat\s+\((\d+)\))?",
            r"[Bb]ab\s+([IVX]+|\d+)",
            r"[Uu]ndang-[Uu]ndang\s+[Nn]omor\s+(\d+)\s+[Tt]ahun\s+(\d+)",
            r"[Pp]eraturan\s+[Pp]emerintah\s+[Nn]omor\s+(\d+)\s+[Tt]ahun\s+(\d+)",
            r"[Pp]eraturan\s+[Mm]enteri\s+[Nn]omor\s+(\d+)\s+[Tt]ahun\s+(\d+)",
            r"[Kk]eputusan\s+[Pp]residen\s+[Nn]omor\s+(\d+)\s+[Tt]ahun\s+(\d+)",
            r"sebagaimana\s+(?:dimaksud|diatur)\s+dalam\s+([^,\.]+)",
            r"berdasarkan\s+([^,\.]+)",
            r"sesuai\s+dengan\s+([^,\.]+)"
        ]

    async def analyze_document(
        self,
        document_id: str,
        document_title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentIntelligence:
        """
        Perform comprehensive document intelligence analysis.

        Args:
            document_id: Unique document identifier
            document_title: Document title
            content: Full document content
            metadata: Additional document metadata

        Returns:
            Complete document intelligence analysis
        """
        try:
            logger.info(f"Starting document intelligence analysis for: {document_title}")

            # Extract legal concepts
            legal_concepts = await self._extract_legal_concepts(content)

            # Analyze document structure
            legal_structure = self._analyze_legal_structure(content)

            # Extract key provisions
            key_provisions = self._extract_key_provisions(content, legal_concepts)

            # Build citation network
            citation_network = self._build_citation_network(content)

            # Analyze document relationships (would need other documents for full analysis)
            document_relationships = self._analyze_document_relationships(content, document_title)

            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                legal_concepts, legal_structure, citation_network
            )

            analysis = DocumentIntelligence(
                document_id=document_id,
                document_title=document_title,
                legal_concepts=legal_concepts,
                document_relationships=document_relationships,
                citation_network=citation_network,
                legal_structure=legal_structure,
                key_provisions=key_provisions,
                complexity_score=complexity_score,
                analysis_timestamp=datetime.now().isoformat()
            )

            logger.info(f"Document intelligence analysis completed: {len(legal_concepts)} concepts extracted")
            return analysis

        except Exception as e:
            logger.error(f"Document intelligence analysis failed: {e}")
            return self._generate_fallback_analysis(document_id, document_title, content)

    async def _extract_legal_concepts(self, content: str) -> List[LegalConcept]:
        """Extract legal concepts using AI and pattern matching."""
        concepts = {}

        # Pattern-based extraction
        for category, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        term = match[0] if match[0] else match[1]
                    else:
                        term = match

                    term = term.strip()
                    if len(term) > 3:
                        if term not in concepts:
                            concepts[term] = {
                                "category": category,
                                "frequency": 0,
                                "contexts": []
                            }
                        concepts[term]["frequency"] += 1

        # AI-enhanced concept extraction
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "test_key":
            ai_concepts = await self._extract_concepts_with_ai(content)
            for concept in ai_concepts:
                if concept["term"] not in concepts:
                    concepts[concept["term"]] = concept

        # Convert to LegalConcept objects
        legal_concepts = []
        for term, data in concepts.items():
            # Get related terms
            related_terms = self._find_related_terms(term, content)

            # Calculate importance score
            importance_score = self._calculate_importance_score(
                data["frequency"], len(content), data["category"]
            )

            # Get context examples
            context_examples = self._extract_context_examples(term, content)

            legal_concept = LegalConcept(
                term=term,
                category=data.get("category", "general"),
                definition=data.get("definition", ""),
                frequency=data["frequency"],
                importance_score=importance_score,
                context_examples=context_examples[:3],
                related_terms=related_terms[:5]
            )
            legal_concepts.append(legal_concept)

        # Sort by importance
        legal_concepts.sort(key=lambda x: x.importance_score, reverse=True)
        return legal_concepts[:50]  # Top 50 concepts

    async def _extract_concepts_with_ai(self, content: str) -> List[Dict[str, Any]]:
        """Extract legal concepts using AI."""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Limit content for AI processing
            content_sample = content[:3000] if len(content) > 3000 else content

            prompt = f"""
            Analisis dokumen hukum berikut dan ekstrak konsep-konsep hukum penting:

            {content_sample}

            Tugas:
            1. Identifikasi maksimal 20 konsep hukum yang paling penting
            2. Untuk setiap konsep, berikan kategori (definisi, kewajiban, hak, larangan, sanksi, prosedur)
            3. Berikan definisi singkat jika memungkinkan

            Format output JSON:
            [{{"term": "konsep", "category": "kategori", "definition": "definisi"}}]
            """

            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)

            # Parse JSON response
            import json
            try:
                ai_concepts = json.loads(response.text)
                return ai_concepts if isinstance(ai_concepts, list) else []
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI response as JSON")
                return []

        except Exception as e:
            logger.error(f"AI concept extraction failed: {e}")
            return []

    def _find_related_terms(self, term: str, content: str) -> List[str]:
        """Find terms related to the given term."""
        term_lower = term.lower()
        related_terms = set()

        # Find terms in the same category
        for category, terms in self.concept_categories.items():
            if any(t in term_lower for t in terms):
                related_terms.update([t for t in terms if t != term_lower])

        # Find co-occurring terms
        sentences = re.split(r'[.!?]', content)
        for sentence in sentences:
            if term_lower in sentence.lower():
                words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
                related_terms.update(words)

        # Remove the original term and limit results
        related_terms.discard(term_lower)
        return list(related_terms)[:5]

    def _calculate_importance_score(self, frequency: int, doc_length: int, category: str) -> float:
        """Calculate importance score for a legal concept."""
        # Base score from frequency
        freq_score = min(1.0, frequency / 10.0)

        # Category weight
        category_weights = {
            "definitions": 1.0,
            "obligations": 0.9,
            "rights": 0.9,
            "sanctions": 0.8,
            "prohibitions": 0.8,
            "procedures": 0.7
        }
        category_weight = category_weights.get(category, 0.5)

        # Document length normalization
        length_factor = min(1.0, doc_length / 10000)

        return freq_score * category_weight * length_factor

    def _extract_context_examples(self, term: str, content: str) -> List[str]:
        """Extract context examples for a term."""
        examples = []
        sentences = re.split(r'[.!?]', content)

        for sentence in sentences:
            if term.lower() in sentence.lower():
                cleaned_sentence = sentence.strip()
                if 20 < len(cleaned_sentence) < 200:
                    examples.append(cleaned_sentence)

        return examples

    def _analyze_legal_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the legal structure of the document."""
        structure = {
            "chapters": [],
            "articles": [],
            "sections": [],
            "hierarchy_depth": 0,
            "organization_pattern": "unknown"
        }

        # Extract chapters (BAB)
        chapter_pattern = r'BAB\s+([IVX]+|\d+)\s*([^\n]+)'
        chapters = re.findall(chapter_pattern, content, re.IGNORECASE)
        structure["chapters"] = [{"number": ch[0], "title": ch[1].strip()} for ch in chapters]

        # Extract articles (Pasal)
        article_pattern = r'Pasal\s+(\d+)(?:\s*([^\n]+))?'
        articles = re.findall(article_pattern, content, re.IGNORECASE)
        structure["articles"] = [{"number": art[0], "title": art[1].strip() if art[1] else ""} for art in articles]

        # Calculate hierarchy depth
        structure["hierarchy_depth"] = self._calculate_hierarchy_depth(content)

        # Determine organization pattern
        structure["organization_pattern"] = self._determine_organization_pattern(structure)

        return structure

    def _calculate_hierarchy_depth(self, content: str) -> int:
        """Calculate the hierarchical depth of the document."""
        patterns = [
            r'BAB\s+[IVX]+',  # Chapter level
            r'Pasal\s+\d+',   # Article level
            r'\(\d+\)',       # Paragraph level
            r'[a-z]\.',       # Sub-paragraph level
            r'\d+\)',         # Item level
        ]

        depth = 0
        for pattern in patterns:
            if re.search(pattern, content):
                depth += 1

        return depth

    def _determine_organization_pattern(self, structure: Dict[str, Any]) -> str:
        """Determine the organizational pattern of the document."""
        if structure["chapters"] and structure["articles"]:
            return "chapter_article_structure"
        elif structure["articles"]:
            return "article_based_structure"
        elif len(structure["chapters"]) > 5:
            return "chapter_heavy_structure"
        else:
            return "simple_structure"

    def _extract_key_provisions(
        self,
        content: str,
        legal_concepts: List[LegalConcept]
    ) -> List[Dict[str, Any]]:
        """Extract key provisions from the document."""
        provisions = []

        # Extract provisions based on legal concepts
        high_importance_concepts = [c for c in legal_concepts if c.importance_score > 0.7]

        for concept in high_importance_concepts[:10]:
            # Find sentences containing this concept
            sentences = re.split(r'[.!?]', content)
            for sentence in sentences:
                if concept.term.lower() in sentence.lower():
                    provision = {
                        "content": sentence.strip(),
                        "related_concept": concept.term,
                        "concept_category": concept.category,
                        "importance_score": concept.importance_score,
                        "provision_type": self._classify_provision_type(sentence)
                    }
                    provisions.append(provision)

        # Remove duplicates and sort by importance
        unique_provisions = []
        seen_content = set()
        for prov in provisions:
            if prov["content"] not in seen_content and len(prov["content"]) > 50:
                unique_provisions.append(prov)
                seen_content.add(prov["content"])

        return sorted(unique_provisions, key=lambda x: x["importance_score"], reverse=True)[:20]

    def _classify_provision_type(self, sentence: str) -> str:
        """Classify the type of legal provision."""
        sentence_lower = sentence.lower()

        if any(word in sentence_lower for word in ["definisi", "yang dimaksud", "pengertian"]):
            return "definition"
        elif any(word in sentence_lower for word in ["wajib", "harus", "berkewajiban"]):
            return "obligation"
        elif any(word in sentence_lower for word in ["berhak", "dapat", "diperbolehkan"]):
            return "right"
        elif any(word in sentence_lower for word in ["dilarang", "tidak boleh", "tidak dapat"]):
            return "prohibition"
        elif any(word in sentence_lower for word in ["sanksi", "pidana", "denda", "hukuman"]):
            return "sanction"
        elif any(word in sentence_lower for word in ["prosedur", "tata cara", "mekanisme"]):
            return "procedure"
        else:
            return "general"

    def _build_citation_network(self, content: str) -> CitationNetwork:
        """Build citation network from the document."""
        citations = []
        internal_citations = 0
        external_citations = 0

        # Extract citations using patterns
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)

        # Count citation types
        for citation in citations:
            citation_str = str(citation).lower()
            if "pasal" in citation_str or "bab" in citation_str:
                internal_citations += 1
            else:
                external_citations += 1

        # Find most cited provisions
        citation_counter = Counter(str(c) for c in citations)
        most_cited = [
            {"provision": prov, "count": count}
            for prov, count in citation_counter.most_common(10)
        ]

        # Simple clustering (would be more sophisticated in practice)
        citation_clusters = self._create_citation_clusters(citations)

        # Calculate authority score
        authority_score = self._calculate_authority_score(internal_citations, external_citations)

        return CitationNetwork(
            total_citations=len(citations),
            internal_citations=internal_citations,
            external_citations=external_citations,
            most_cited_provisions=most_cited,
            citation_clusters=citation_clusters,
            authority_score=authority_score
        )

    def _create_citation_clusters(self, citations: List) -> List[Dict[str, Any]]:
        """Create citation clusters for network analysis."""
        # Simplified clustering based on citation types
        clusters = defaultdict(list)

        for citation in citations:
            citation_str = str(citation).lower()
            if "pasal" in citation_str:
                clusters["articles"].append(citation_str)
            elif "bab" in citation_str:
                clusters["chapters"].append(citation_str)
            elif "undang-undang" in citation_str:
                clusters["laws"].append(citation_str)
            else:
                clusters["others"].append(citation_str)

        return [
            {"type": cluster_type, "citations": citations, "count": len(citations)}
            for cluster_type, citations in clusters.items()
        ]

    def _calculate_authority_score(self, internal_citations: int, external_citations: int) -> float:
        """Calculate authority score based on citation patterns."""
        total_citations = internal_citations + external_citations
        if total_citations == 0:
            return 0.0

        # Balance between internal structure and external references
        internal_ratio = internal_citations / total_citations
        external_ratio = external_citations / total_citations

        # Higher score for documents with good internal structure and external references
        authority_score = (internal_ratio * 0.6 + external_ratio * 0.4) * min(1.0, total_citations / 20)

        return min(1.0, authority_score)

    def _analyze_document_relationships(
        self,
        content: str,
        document_title: str
    ) -> List[DocumentRelationship]:
        """Analyze relationships with other documents."""
        relationships = []

        # Extract references to other documents
        doc_patterns = [
            r'[Uu]ndang-[Uu]ndang\s+[Nn]omor\s+\d+\s+[Tt]ahun\s+\d+[^,\.]*',
            r'[Pp]eraturan\s+[Pp]emerintah\s+[Nn]omor\s+\d+\s+[Tt]ahun\s+\d+[^,\.]*',
            r'[Pp]eraturan\s+[Mm]enteri\s+[Nn]omor\s+\d+\s+[Tt]ahun\s+\d+[^,\.]*'
        ]

        referenced_docs = set()
        for pattern in doc_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            referenced_docs.update(matches)

        # Create relationships
        for doc in referenced_docs:
            relationship = DocumentRelationship(
                source_document=document_title,
                target_document=doc.strip(),
                relationship_type="references",
                strength=0.8,
                evidence=[f"Referenced in: {document_title}"],
                description=f"Document {document_title} references {doc}"
            )
            relationships.append(relationship)

        return relationships[:20]  # Limit to top 20

    def _calculate_complexity_score(
        self,
        legal_concepts: List[LegalConcept],
        legal_structure: Dict[str, Any],
        citation_network: CitationNetwork
    ) -> float:
        """Calculate document complexity score."""
        # Base complexity from concepts
        concept_complexity = min(1.0, len(legal_concepts) / 50)

        # Structure complexity
        structure_complexity = min(1.0, legal_structure["hierarchy_depth"] / 5)

        # Citation complexity
        citation_complexity = min(1.0, citation_network.total_citations / 30)

        # Average with weights
        complexity_score = (
            concept_complexity * 0.4 +
            structure_complexity * 0.3 +
            citation_complexity * 0.3
        )

        return complexity_score

    def _generate_fallback_analysis(
        self,
        document_id: str,
        document_title: str,
        content: str
    ) -> DocumentIntelligence:
        """Generate fallback analysis when full processing fails."""
        return DocumentIntelligence(
            document_id=document_id,
            document_title=document_title,
            legal_concepts=[],
            document_relationships=[],
            citation_network=CitationNetwork(
                total_citations=0,
                internal_citations=0,
                external_citations=0,
                most_cited_provisions=[],
                citation_clusters=[],
                authority_score=0.0
            ),
            legal_structure={"chapters": [], "articles": [], "hierarchy_depth": 0},
            key_provisions=[],
            complexity_score=0.0,
            analysis_timestamp=datetime.now().isoformat()
        )

    async def bulk_analyze_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentIntelligence]:
        """Analyze multiple documents in batch."""
        analyses = []

        for doc in documents:
            try:
                analysis = await self.analyze_document(
                    document_id=doc["id"],
                    document_title=doc["title"],
                    content=doc["content"],
                    metadata=doc.get("metadata")
                )
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze document {doc['id']}: {e}")

        return analyses

    def build_concept_network(self, analyses: List[DocumentIntelligence]) -> Dict[str, Any]:
        """Build concept network across multiple documents."""
        # Create networkx graph
        G = nx.Graph()

        # Add nodes (concepts)
        all_concepts = {}
        for analysis in analyses:
            for concept in analysis.legal_concepts:
                if concept.term not in all_concepts:
                    all_concepts[concept.term] = {
                        "category": concept.category,
                        "total_frequency": 0,
                        "documents": []
                    }
                all_concepts[concept.term]["total_frequency"] += concept.frequency
                all_concepts[concept.term]["documents"].append(analysis.document_title)

        # Add nodes to graph
        for term, data in all_concepts.items():
            G.add_node(term, **data)

        # Add edges (relationships)
        for analysis in analyses:
            concepts_in_doc = [c.term for c in analysis.legal_concepts]
            for i, concept1 in enumerate(concepts_in_doc):
                for concept2 in concepts_in_doc[i+1:]:
                    if G.has_edge(concept1, concept2):
                        G[concept1][concept2]["weight"] += 1
                    else:
                        G.add_edge(concept1, concept2, weight=1)

        # Calculate network metrics
        centrality = nx.degree_centrality(G)
        clustering = nx.clustering(G)

        return {
            "total_concepts": len(all_concepts),
            "total_relationships": G.number_of_edges(),
            "most_central_concepts": sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10],
            "average_clustering": sum(clustering.values()) / len(clustering) if clustering else 0,
            "concept_categories": Counter(data["category"] for data in all_concepts.values())
        }

    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get statistics about document intelligence capabilities."""
        return {
            "pattern_categories": len(self.legal_patterns),
            "total_patterns": sum(len(patterns) for patterns in self.legal_patterns.values()),
            "concept_categories": len(self.concept_categories),
            "citation_patterns": len(self.citation_patterns),
            "analysis_features": [
                "Legal concept extraction",
                "Document structure analysis",
                "Citation network building",
                "Relationship analysis",
                "Complexity scoring",
                "AI-enhanced extraction",
                "Cross-document analysis"
            ],
            "supported_extractions": [
                "Definitions and terminology",
                "Rights and obligations",
                "Procedures and processes",
                "Sanctions and penalties",
                "Legal entities and roles",
                "Citations and references",
                "Document hierarchy"
            ]
        }
