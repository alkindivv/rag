#!/usr/bin/env python3
"""
Comprehensive Accuracy Fixes Implementation for Legal RAG System

This module implements targeted fixes for accuracy issues identified in the comprehensive audit:
1. Comparative query processing improvements
2. False positive filtering
3. Content quality scoring
4. Confidence calibration
5. Enhanced legal domain understanding

Author: Senior AI Systems Auditor and Search Accuracy Specialist
Purpose: Fix critical accuracy issues and eliminate false positives
"""

import re
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.search.vector_search import VectorSearchService, SearchResult
from src.services.llm.legal_llm import LegalLLMService
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContentQualityScore:
    """Content quality scoring result."""
    score: float  # 0.0-1.0
    is_generic: bool
    is_relevant: bool
    quality_issues: List[str]
    recommendations: List[str]


class ComparativeQueryProcessor:
    """Enhanced processor for comparative legal queries."""

    def __init__(self):
        """Initialize comparative query processor."""
        self.comparative_patterns = self._build_comparative_patterns()
        self.legal_concept_map = self._build_legal_concept_map()

    def _build_comparative_patterns(self) -> List[re.Pattern]:
        """Build patterns to detect comparative queries."""
        patterns = [
            re.compile(r'\b(?:apa\s+)?(?:bedanya|perbedaan)\s+(?:antara\s+)?(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)', re.IGNORECASE),
            re.compile(r'\b(?:beda|berbeda)\s+(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)', re.IGNORECASE),
            re.compile(r'\b(?:perbandingan|membandingkan)\s+(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)', re.IGNORECASE),
            re.compile(r'\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)', re.IGNORECASE),
        ]
        return patterns

    def _build_legal_concept_map(self) -> Dict[str, Dict[str, str]]:
        """Build mapping of legal concepts for better comparative analysis."""
        return {
            'pembunuhan': {
                'sengaja': 'pembunuhan dengan maksud',
                'tidak_sengaja': 'pembunuhan karena kelalaian',
                'berencana': 'pembunuhan berencana',
                'biasa': 'pembunuhan tanpa pemberatan'
            },
            'sanksi': {
                'pidana': 'sanksi pidana penjara denda',
                'administratif': 'sanksi administratif teguran pencabutan izin',
                'perdata': 'sanksi perdata ganti rugi'
            },
            'hukuman': {
                'ringan': 'hukuman ringan teguran',
                'sedang': 'hukuman sedang penjara',
                'berat': 'hukuman berat penjara lama'
            }
        }

    def detect_comparative_query(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect if query is comparative and extract components."""
        for pattern in self.comparative_patterns:
            match = pattern.search(query)
            if match:
                concept_a = match.group(1).strip()
                concept_b = match.group(2).strip()
                return (concept_a, concept_b)
        return None

    def enhance_comparative_query(self, query: str, concept_a: str, concept_b: str) -> List[str]:
        """Generate enhanced queries for comparative analysis."""
        enhanced_queries = []

        # Base comparative query
        enhanced_queries.append(query)

        # Individual concept queries
        enhanced_queries.append(f"definisi atau pengertian {concept_a}")
        enhanced_queries.append(f"definisi atau pengertian {concept_b}")

        # Specific legal provision queries
        enhanced_queries.append(f"ketentuan hukum mengenai {concept_a}")
        enhanced_queries.append(f"ketentuan hukum mengenai {concept_b}")

        # Enhanced with legal context
        for base_concept, expansions in self.legal_concept_map.items():
            if base_concept.lower() in query.lower():
                for key, expansion in expansions.items():
                    if key.lower() in concept_a.lower():
                        enhanced_queries.append(f"{expansion} dalam hukum indonesia")
                    if key.lower() in concept_b.lower():
                        enhanced_queries.append(f"{expansion} dalam hukum indonesia")

        return enhanced_queries[:6]  # Limit to avoid too many queries


class FalsePositiveFilter:
    """Filter to eliminate false positive patterns."""

    def __init__(self):
        """Initialize false positive filter."""
        self.generic_patterns = self._build_generic_patterns()
        self.irrelevant_content_patterns = self._build_irrelevant_patterns()
        self.quality_thresholds = {
            'min_content_length': 50,
            'max_generic_ratio': 0.3,
            'min_keyword_density': 0.02
        }

    def _build_generic_patterns(self) -> List[re.Pattern]:
        """Build patterns for generic content detection."""
        return [
            re.compile(r'\bhuruf\s+[a-z]\b.*(?:keamanan\s+negara|proses\s+kehidupan)', re.IGNORECASE),
            re.compile(r'\bmenarik\s+diri\s+dari\s+kesepakatan', re.IGNORECASE),
            re.compile(r'\btindakan\s+yang\s+patut\s+untuk\s+mencegah', re.IGNORECASE),
            re.compile(r'\btindak\s+pidana\s+di\s+wilayah\s+negara', re.IGNORECASE),
        ]

    def _build_irrelevant_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build patterns for domain-specific irrelevant content."""
        return {
            'criminal_law': [
                re.compile(r'\b(?:kontrak|perjanjian|warisan)\b', re.IGNORECASE),
                re.compile(r'\b(?:perkawinan|perceraian|harta\s+bersama)\b', re.IGNORECASE),
            ],
            'civil_law': [
                re.compile(r'\b(?:pidana|pembunuhan|pencurian)\b', re.IGNORECASE),
                re.compile(r'\b(?:sanksi\s+pidana|hukuman\s+penjara)\b', re.IGNORECASE),
            ],
            'administrative_law': [
                re.compile(r'\b(?:pembunuhan|pencurian|korupsi)\b', re.IGNORECASE),
                re.compile(r'\b(?:perkawinan|warisan)\b', re.IGNORECASE),
            ]
        }

    def evaluate_content_quality(self, result: SearchResult, query_context: str = None) -> ContentQualityScore:
        """Evaluate content quality and detect false positives."""
        content = (result.content or "").strip()
        citation = result.citation_string or ""

        quality_issues = []
        is_generic = False
        is_relevant = True

        # Check for generic patterns
        for pattern in self.generic_patterns:
            if pattern.search(content):
                is_generic = True
                quality_issues.append(f"Generic content pattern detected: {pattern.pattern[:50]}...")

        # Check content length
        if len(content) < self.quality_thresholds['min_content_length']:
            quality_issues.append("Content too short for meaningful analysis")

        # Check for domain relevance if context provided
        if query_context:
            domain_patterns = self.irrelevant_content_patterns.get(query_context, [])
            for pattern in domain_patterns:
                if pattern.search(content):
                    is_relevant = False
                    quality_issues.append(f"Content from wrong legal domain: {pattern.pattern[:30]}...")

        # Check citation quality for "huruf a" generic pattern
        if "huruf a" in citation.lower() and len(content) < 100:
            generic_terms = ["keamanan negara", "menarik diri", "proses kehidupan", "tindak pidana di wilayah"]
            if any(term.lower() in content.lower() for term in generic_terms):
                is_generic = True
                quality_issues.append("Generic 'huruf a' citation with minimal content")

        # Calculate quality score
        base_score = result.score

        # Penalties
        if is_generic:
            base_score *= 0.3
        if not is_relevant:
            base_score *= 0.4
        if len(content) < self.quality_thresholds['min_content_length']:
            base_score *= 0.6

        quality_score = max(0.0, min(1.0, base_score))

        # Generate recommendations
        recommendations = []
        if is_generic:
            recommendations.append("Filter out generic content fragments")
        if not is_relevant:
            recommendations.append("Apply domain-specific relevance filtering")
        if len(quality_issues) > 2:
            recommendations.append("Improve content indexing and chunking strategy")

        return ContentQualityScore(
            score=quality_score,
            is_generic=is_generic,
            is_relevant=is_relevant,
            quality_issues=quality_issues,
            recommendations=recommendations
        )

    def filter_results(self, results: List[SearchResult], query_context: str = None) -> List[SearchResult]:
        """Filter results to remove false positives."""
        filtered_results = []

        for result in results:
            quality = self.evaluate_content_quality(result, query_context)

            # Keep result if quality is acceptable
            if quality.score >= 0.5 and not quality.is_generic and quality.is_relevant:
                filtered_results.append(result)
            elif quality.score >= 0.7:  # High score overrides other concerns
                filtered_results.append(result)

        return filtered_results


class ConfidenceCalibrator:
    """Calibrate confidence scores for better precision."""

    def __init__(self):
        """Initialize confidence calibrator."""
        self.calibration_factors = {
            'citation_exact_match': 1.0,
            'semantic_similarity': 0.8,
            'keyword_match': 0.6,
            'domain_relevance': 0.9,
            'content_quality': 0.7
        }

    def recalibrate_confidence(self, result: SearchResult, query: str, expected_pasal: str = None) -> float:
        """Recalibrate confidence score based on multiple factors."""
        base_score = result.score
        citation = result.citation_string or ""
        content = (result.content or "").lower()
        query_lower = query.lower()

        # Factor 1: Citation exact match
        citation_factor = 1.0
        if expected_pasal and expected_pasal in citation:
            citation_factor = self.calibration_factors['citation_exact_match']
        elif "pasal" in query_lower and "pasal" in citation.lower():
            citation_factor = 0.9
        else:
            citation_factor = 0.8

        # Factor 2: Semantic similarity (use base score)
        semantic_factor = self.calibration_factors['semantic_similarity']

        # Factor 3: Keyword match
        query_keywords = re.findall(r'\b\w+\b', query_lower)
        legal_keywords = [kw for kw in query_keywords if len(kw) > 3 and kw not in {'yang', 'untuk', 'dengan', 'dalam', 'pada', 'dari'}]

        if legal_keywords:
            keyword_matches = sum(1 for kw in legal_keywords if kw in content)
            keyword_ratio = keyword_matches / len(legal_keywords)
            keyword_factor = 0.5 + (keyword_ratio * 0.5)  # 0.5 to 1.0
        else:
            keyword_factor = self.calibration_factors['keyword_match']

        # Factor 4: Content quality (check length and specificity)
        if len(content) > 100 and not re.search(r'\bhuruf\s+[a-z]\b.*(?:keamanan|menarik\s+diri)', content):
            quality_factor = self.calibration_factors['content_quality']
        else:
            quality_factor = 0.4

        # Combined confidence
        calibrated_score = base_score * citation_factor * semantic_factor * keyword_factor * quality_factor

        return min(1.0, calibrated_score)


class EnhancedLegalRAGService:
    """Enhanced Legal RAG service with accuracy fixes."""

    def __init__(self):
        """Initialize enhanced service with accuracy components."""
        self.search_service = VectorSearchService()
        self.llm_service = LegalLLMService()
        self.comparative_processor = ComparativeQueryProcessor()
        self.false_positive_filter = FalsePositiveFilter()
        self.confidence_calibrator = ConfidenceCalibrator()

    async def enhanced_search(
        self,
        query: str,
        k: int = 5,
        use_reranking: bool = True,
        filter_false_positives: bool = True
    ) -> Dict[str, Any]:
        """Enhanced search with accuracy fixes."""
        start_time = time.time()

        # Step 1: Detect comparative queries
        comparative_components = self.comparative_processor.detect_comparative_query(query)

        if comparative_components:
            logger.info(f"Detected comparative query: {comparative_components}")
            results = await self._handle_comparative_query(query, comparative_components, k)
        else:
            # Standard search
            search_result = await self.search_service.search_async(
                query=query,
                k=k * 2,  # Get more results for filtering
                use_reranking=use_reranking
            )
            results = search_result["results"]

        # Step 2: Apply false positive filtering
        if filter_false_positives:
            query_domain = self._detect_query_domain(query)
            results = self.false_positive_filter.filter_results(results, query_domain)

        # Step 3: Recalibrate confidence scores
        expected_pasal = self._extract_expected_pasal(query)
        for result in results:
            new_confidence = self.confidence_calibrator.recalibrate_confidence(
                result, query, expected_pasal
            )
            result.score = new_confidence

        # Step 4: Re-sort by new confidence and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:k]

        duration_ms = (time.time() - start_time) * 1000

        return {
            "results": results,
            "metadata": {
                "query": query,
                "search_type": "enhanced_accuracy",
                "total_results": len(results),
                "duration_ms": round(duration_ms, 2),
                "comparative_query": comparative_components is not None,
                "false_positive_filtering": filter_false_positives
            }
        }

    async def _handle_comparative_query(
        self,
        original_query: str,
        components: Tuple[str, str],
        k: int
    ) -> List[SearchResult]:
        """Handle comparative queries with enhanced processing."""
        concept_a, concept_b = components

        # Generate enhanced queries
        enhanced_queries = self.comparative_processor.enhance_comparative_query(
            original_query, concept_a, concept_b
        )

        all_results = []
        seen_citations = set()

        # Execute multiple searches concurrently
        search_tasks = []
        for enhanced_query in enhanced_queries:
            task = self.search_service.search_async(
                query=enhanced_query,
                k=3,
                use_reranking=True
            )
            search_tasks.append(task)

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Combine and deduplicate results
        for search_result in search_results:
            if isinstance(search_result, dict) and "results" in search_result:
                for result in search_result["results"]:
                    citation = result.citation_string or ""
                    if citation not in seen_citations:
                        seen_citations.add(citation)
                        all_results.append(result)

        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k * 2]  # Return more for further filtering

    def _detect_query_domain(self, query: str) -> str:
        """Detect legal domain of query for relevance filtering."""
        query_lower = query.lower()

        criminal_keywords = ['pidana', 'pembunuhan', 'pencurian', 'korupsi', 'hukuman', 'sanksi pidana']
        civil_keywords = ['kontrak', 'perkawinan', 'warisan', 'harta', 'perdata', 'ganti rugi']
        admin_keywords = ['izin', 'administratif', 'perizinan', 'sanksi administratif', 'tata usaha']

        if any(kw in query_lower for kw in criminal_keywords):
            return 'criminal_law'
        elif any(kw in query_lower for kw in civil_keywords):
            return 'civil_law'
        elif any(kw in query_lower for kw in admin_keywords):
            return 'administrative_law'
        else:
            return 'general'

    def _extract_expected_pasal(self, query: str) -> Optional[str]:
        """Extract expected pasal number from query."""
        pasal_match = re.search(r'\bpasal\s+(\d+)', query, re.IGNORECASE)
        if pasal_match:
            return pasal_match.group(1)
        return None

    async def enhanced_answer_generation(
        self,
        query: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate answer with enhanced accuracy."""
        # Use enhanced search
        search_results = await self.enhanced_search(query, k=5)

        # Generate answer with quality-filtered context
        answer = await self.llm_service.generate_answer(
            query=query,
            context=search_results["results"],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add accuracy metadata
        answer["enhanced_accuracy"] = {
            "false_positive_filtering": True,
            "confidence_calibration": True,
            "comparative_processing": search_results["metadata"]["comparative_query"]
        }

        return answer


async def test_accuracy_fixes():
    """Test the accuracy fixes with problematic queries."""

    service = EnhancedLegalRAGService()

    test_queries = [
        "apa bedanya hukuman bagi orang yang melakukan pembunuhan tanpa sengaja dengan hukuman bagi orang yang melakukan pembunuhan berencana?",
        "berapa lama hukuman untuk pembunuhan?",
        "perbedaan sanksi pidana dan sanksi administratif?",
        "pasal 458 UU 1 tahun 2023"
    ]

    print("🔧 TESTING ACCURACY FIXES")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Testing: {query}")

        # Test enhanced search
        result = await service.enhanced_search(query)

        print(f"Results: {len(result['results'])}")
        print(f"Enhanced: {result['metadata']['search_type']}")

        for j, res in enumerate(result['results'][:3]):
            print(f"  {j+1}. Score: {res.score:.3f} | {res.citation_string}")

    print("\n✅ Accuracy fixes testing completed!")


if __name__ == "__main__":
    asyncio.run(test_accuracy_fixes())
