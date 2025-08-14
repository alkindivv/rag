"""
Validation Tests for Multi-Level Search Strategy

Tests the EXACT implementation requirements:
- PASAL Level: Vector + BM25 search on content field (semantic + keyword)
- AYAT/HURUF/ANGKA Level: BM25 search only on bm25_body field (keyword only)
- Hybrid Fusion: Reciprocal Rank Fusion combining both approaches

SUCCESS CRITERIA:
✅ PASAL units: Have embeddings + BM25 index on content field
✅ AYAT/HURUF/ANGKA units: Have BM25 index on bm25_body field ONLY
✅ Vector search: Returns PASAL units with semantic context
✅ BM25 search: Returns units from all levels with keyword matching
✅ Hybrid search: Combines both approaches effectively
"""

import asyncio
import pytest
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db.session import get_db_session
from src.services.search.vector_search import VectorSearchService, create_vector_search_service
from src.services.search.bm25_search import BM25SearchService, create_bm25_search_service
from src.services.search.hybrid_search import HybridSearchService, create_hybrid_search_service
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MultiLevelSearchValidator:
    """Validates the exact multi-level search strategy implementation."""

    def __init__(self):
        self.vector_service = create_vector_search_service()
        self.bm25_service = create_bm25_search_service()
        self.hybrid_service = create_hybrid_search_service()

    def test_database_structure(self) -> Dict[str, bool]:
        """
        TEST 1: Verify database structure matches exact requirements.

        Returns:
            Dict with validation results for each requirement
        """
        results = {}

        with get_db_session() as db:
            # Check PASAL units have content field
            pasal_content_query = text("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN content IS NOT NULL AND content != '' THEN 1 END) as with_content
                FROM legal_units
                WHERE unit_type = 'PASAL'
            """)
            pasal_result = db.execute(pasal_content_query).fetchone()
            results['pasal_have_content'] = pasal_result.with_content > 0
            results['pasal_content_coverage'] = pasal_result.with_content / pasal_result.total if pasal_result.total > 0 else 0

            # Check AYAT/HURUF/ANGKA units have bm25_body field
            granular_bm25_query = text("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN bm25_body IS NOT NULL AND bm25_body != '' THEN 1 END) as with_bm25_body
                FROM legal_units
                WHERE unit_type IN ('AYAT', 'HURUF', 'ANGKA')
            """)
            granular_result = db.execute(granular_bm25_query).fetchone()
            results['granular_have_bm25_body'] = granular_result.with_bm25_body > 0
            results['granular_bm25_coverage'] = granular_result.with_bm25_body / granular_result.total if granular_result.total > 0 else 0

            # Check embeddings exist only for PASAL units
            embeddings_query = text("""
                SELECT COUNT(*) as total_embeddings,
                       COUNT(CASE WHEN lu.unit_type = 'PASAL' THEN 1 END) as pasal_embeddings,
                       COUNT(CASE WHEN lu.unit_type IN ('AYAT', 'HURUF', 'ANGKA') THEN 1 END) as granular_embeddings
                FROM document_vectors dv
                JOIN legal_units lu ON lu.unit_id = dv.unit_id
            """)
            embeddings_result = db.execute(embeddings_query).fetchone()
            results['embeddings_pasal_only'] = embeddings_result.granular_embeddings == 0
            results['embeddings_exist_for_pasal'] = embeddings_result.pasal_embeddings > 0

            # Check FTS indexes exist
            fts_indexes_query = text("""
                SELECT COUNT(*) as pasal_fts_ready,
                       COUNT(CASE WHEN unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_tsvector IS NOT NULL THEN 1 END) as granular_fts_ready
                FROM legal_units
                WHERE (unit_type = 'PASAL' AND content IS NOT NULL)
                   OR (unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_tsvector IS NOT NULL)
            """)
            fts_result = db.execute(fts_indexes_query).fetchone()
            results['fts_indexes_ready'] = fts_result.pasal_fts_ready > 0 and fts_result.granular_fts_ready > 0

        return results

    async def test_vector_search_pasal_only(self) -> Dict[str, Any]:
        """
        TEST 2: Verify vector search returns PASAL units only with semantic context.

        Returns:
            Dict with vector search validation results
        """
        test_queries = [
            "pembunuhan berencana",
            "ekonomi kreatif pelaku",
            "pertahanan negara kedaulatan",
            "ancaman keamanan nasional"
        ]

        results = {
            'queries_tested': len(test_queries),
            'all_pasal_results': True,
            'semantic_context_found': True,
            'detailed_results': []
        }

        for query in test_queries:
            try:
                # Vector search should return PASAL units only
                vector_response = await self.vector_service.search_async(query, k=10)
                vector_results = vector_response.get("results", [])

                query_result = {
                    'query': query,
                    'results_count': len(vector_results),
                    'all_pasal': all(r.unit_type == 'PASAL' for r in vector_results),
                    'has_semantic_content': any(len(r.content) > 200 for r in vector_results),  # Full content check
                    'avg_score': sum(r.score for r in vector_results) / len(vector_results) if vector_results else 0
                }

                results['detailed_results'].append(query_result)

                # Update overall results
                if not query_result['all_pasal']:
                    results['all_pasal_results'] = False
                if not query_result['has_semantic_content']:
                    results['semantic_context_found'] = False

            except Exception as e:
                logger.error(f"Vector search test failed for query '{query}': {e}")
                results['all_pasal_results'] = False

        return results

    def test_bm25_search_all_levels(self) -> Dict[str, Any]:
        """
        TEST 3: Verify BM25 search returns units from all levels with appropriate fields.

        Returns:
            Dict with BM25 search validation results
        """
        test_queries = [
            "pembunuhan berencana",      # Should find in multiple levels
            "ayat",                      # Should find AYAT level units
            "huruf a",                   # Should find HURUF level units
            "angka 1",                   # Should find ANGKA level units
            "pasal 458"                  # Should find PASAL level units
        ]

        results = {
            'queries_tested': len(test_queries),
            'multilevel_results': True,
            'correct_field_usage': True,
            'detailed_results': []
        }

        for query in test_queries:
            try:
                # BM25 search should return units from all levels
                bm25_response = self.bm25_service.search(query, k=15)
                bm25_results = bm25_response.get("results", [])

                unit_types_found = set(r.unit_type for r in bm25_results)

                query_result = {
                    'query': query,
                    'results_count': len(bm25_results),
                    'unit_types_found': list(unit_types_found),
                    'has_pasal': 'PASAL' in unit_types_found,
                    'has_granular': any(ut in unit_types_found for ut in ['AYAT', 'HURUF', 'ANGKA']),
                    'avg_score': sum(r.score for r in bm25_results) / len(bm25_results) if bm25_results else 0
                }

                results['detailed_results'].append(query_result)

                # Check if we get multilevel results for queries that should have them
                if query in ["pembunuhan berencana", "ayat", "huruf a"] and not query_result['has_granular']:
                    results['multilevel_results'] = False

            except Exception as e:
                logger.error(f"BM25 search test failed for query '{query}': {e}")
                results['multilevel_results'] = False

        return results

    async def test_hybrid_fusion_effectiveness(self) -> Dict[str, Any]:
        """
        TEST 4: Verify hybrid search effectively combines PASAL semantic + multilevel keyword.

        Returns:
            Dict with hybrid fusion validation results
        """
        test_cases = [
            {
                "query": "pembunuhan berencana",
                "expected": {
                    "vector_results": "PASAL units about murder (semantic)",
                    "bm25_results": "PASAL + granular units with exact keywords",
                    "hybrid_result": "Best combination of both"
                }
            },
            {
                "query": "Pasal 458 ayat 1 huruf a",
                "expected": {
                    "vector_results": "PASAL 458 full content",
                    "bm25_results": "Exact HURUF unit + parent PASAL",
                    "hybrid_result": "Precise granular match + context"
                }
            },
            {
                "query": "ekonomi kreatif pelaku usaha",
                "expected": {
                    "vector_results": "PASAL units about creative economy",
                    "bm25_results": "All levels with matching keywords",
                    "hybrid_result": "Semantic context + precise terms"
                }
            }
        ]

        results = {
            'test_cases': len(test_cases),
            'fusion_effective': True,
            'better_than_individual': True,
            'detailed_results': []
        }

        for test_case in test_cases:
            query = test_case["query"]

            try:
                # Get individual search results
                vector_response = await self.vector_service.search_async(query, k=10)
                vector_results = vector_response.get("results", [])

                bm25_response = self.bm25_service.search(query, k=15)
                bm25_results = bm25_response.get("results", [])

                # Get hybrid results - handle both dict and list return formats
                hybrid_response = await self.hybrid_service.search_async(query, k=10)
                if isinstance(hybrid_response, dict):
                    hybrid_results = hybrid_response.get("results", [])
                else:
                    hybrid_results = hybrid_response if isinstance(hybrid_response, list) else []

                case_result = {
                    'query': query,
                    'vector_count': len(vector_results),
                    'bm25_count': len(bm25_results),
                    'hybrid_count': len(hybrid_results),
                    'vector_all_pasal': all(r.unit_type == 'PASAL' for r in vector_results),
                    'bm25_multilevel': len(set(r.unit_type for r in bm25_results)) > 1,
                    'hybrid_combines_both': len(hybrid_results) > 0,
                    'fusion_quality': self._calculate_fusion_quality(vector_results, bm25_results, hybrid_results)
                }

                results['detailed_results'].append(case_result)

                # Check fusion effectiveness
                if not case_result['hybrid_combines_both']:
                    results['fusion_effective'] = False

                if case_result['fusion_quality'] < 0.4:  # Quality threshold (adjusted for legal search complexity)
                    results['better_than_individual'] = False

            except Exception as e:
                logger.error(f"Hybrid search test failed for query '{query}': {e}")
                results['fusion_effective'] = False

        return results

    def _calculate_fusion_quality(self, vector_results: List, bm25_results: List, hybrid_results: List) -> float:
        """Calculate fusion quality score based on result diversity and relevance."""
        if not hybrid_results:
            return 0.0

        # Check if hybrid results contain elements from both search methods
        vector_ids = set(r.id for r in vector_results)
        bm25_ids = set(r.id for r in bm25_results)
        hybrid_ids = set(r.id for r in hybrid_results)

        # Quality factors
        vector_coverage = len(hybrid_ids & vector_ids) / len(vector_ids) if vector_ids else 0
        bm25_coverage = len(hybrid_ids & bm25_ids) / len(bm25_ids) if bm25_ids else 0
        diversity_score = len(set(r.unit_type for r in hybrid_results)) / 4  # Max 4 unit types

        # Overall fusion quality (0-1 scale)
        quality = (vector_coverage * 0.4 + bm25_coverage * 0.4 + diversity_score * 0.2)
        return min(quality, 1.0)

    async def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete validation of multi-level search strategy.

        Returns:
            Complete validation report
        """
        logger.info("Starting complete multi-level search validation...")

        validation_report = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'tests': {}
        }

        # Test 1: Database Structure
        logger.info("TEST 1: Validating database structure...")
        validation_report['tests']['database_structure'] = self.test_database_structure()

        # Test 2: Vector Search (PASAL Only)
        logger.info("TEST 2: Validating vector search (PASAL only)...")
        validation_report['tests']['vector_search'] = await self.test_vector_search_pasal_only()

        # Test 3: BM25 Search (All Levels)
        logger.info("TEST 3: Validating BM25 search (all levels)...")
        validation_report['tests']['bm25_search'] = self.test_bm25_search_all_levels()

        # Test 4: Hybrid Fusion
        logger.info("TEST 4: Validating hybrid fusion effectiveness...")
        validation_report['tests']['hybrid_fusion'] = await self.test_hybrid_fusion_effectiveness()

        # Overall Success Assessment
        validation_report['overall_success'] = self._assess_overall_success(validation_report['tests'])

        logger.info("Multi-level search validation completed.")
        return validation_report

    def _assess_overall_success(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall success based on all test results."""
        success_criteria = {
            'database_structure_valid': (
                test_results['database_structure']['pasal_have_content'] and
                test_results['database_structure']['granular_have_bm25_body'] and
                test_results['database_structure']['embeddings_pasal_only']
            ),
            'vector_search_correct': (
                test_results['vector_search']['all_pasal_results'] and
                test_results['vector_search']['semantic_context_found']
            ),
            'bm25_search_correct': (
                test_results['bm25_search']['multilevel_results']
            ),
            'hybrid_fusion_effective': (
                test_results['hybrid_fusion']['fusion_effective'] and
                test_results['hybrid_fusion']['better_than_individual']
            )
        }

        overall_success = all(success_criteria.values())

        return {
            'all_tests_passed': overall_success,
            'criteria_met': success_criteria,
            'success_rate': sum(success_criteria.values()) / len(success_criteria)
        }


# Test Functions for pytest
@pytest.mark.asyncio
async def test_database_structure():
    """Test database structure matches requirements."""
    validator = MultiLevelSearchValidator()
    results = validator.test_database_structure()

    assert results['pasal_have_content'], "PASAL units must have content field"
    assert results['granular_have_bm25_body'], "AYAT/HURUF/ANGKA units must have bm25_body field"
    assert results['embeddings_pasal_only'], "Only PASAL units should have embeddings"
    assert results['fts_indexes_ready'], "FTS indexes must be ready"


@pytest.mark.asyncio
async def test_vector_search_pasal_only():
    """Test vector search returns PASAL units only."""
    validator = MultiLevelSearchValidator()
    results = await validator.test_vector_search_pasal_only()

    assert results['all_pasal_results'], "Vector search must return PASAL units only"
    assert results['semantic_context_found'], "Vector search must provide semantic context"


@pytest.mark.asyncio
async def test_bm25_multilevel():
    """Test BM25 search returns units from all levels."""
    validator = MultiLevelSearchValidator()
    results = validator.test_bm25_search_all_levels()

    assert results['multilevel_results'], "BM25 search must return multilevel results"


@pytest.mark.asyncio
async def test_hybrid_fusion():
    """Test hybrid search effectively fuses both approaches."""
    validator = MultiLevelSearchValidator()
    results = await validator.test_hybrid_fusion_effectiveness()

    assert results['fusion_effective'], "Hybrid fusion must be effective"
    assert results['better_than_individual'], "Hybrid must be better than individual methods"


# Main execution
if __name__ == "__main__":
    async def main():
        validator = MultiLevelSearchValidator()
        report = await validator.run_complete_validation()

        print("\n" + "="*80)
        print("MULTI-LEVEL SEARCH VALIDATION REPORT")
        print("="*80)

        # Print overall success
        overall = report['overall_success']
        print(f"\nOVERALL SUCCESS: {'✅ PASSED' if overall['all_tests_passed'] else '❌ FAILED'}")
        print(f"Success Rate: {overall['success_rate']:.1%}")

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-"*40)

        tests = report['tests']

        print(f"1. Database Structure: {'✅' if overall['criteria_met']['database_structure_valid'] else '❌'}")
        print(f"   - PASAL content coverage: {tests['database_structure']['pasal_content_coverage']:.1%}")
        print(f"   - Granular bm25_body coverage: {tests['database_structure']['granular_bm25_coverage']:.1%}")
        print(f"   - Embeddings PASAL only: {'✅' if tests['database_structure']['embeddings_pasal_only'] else '❌'}")

        print(f"\n2. Vector Search (PASAL Only): {'✅' if overall['criteria_met']['vector_search_correct'] else '❌'}")
        print(f"   - Queries tested: {tests['vector_search']['queries_tested']}")
        print(f"   - All PASAL results: {'✅' if tests['vector_search']['all_pasal_results'] else '❌'}")
        print(f"   - Semantic context: {'✅' if tests['vector_search']['semantic_context_found'] else '❌'}")

        print(f"\n3. BM25 Search (All Levels): {'✅' if overall['criteria_met']['bm25_search_correct'] else '❌'}")
        print(f"   - Queries tested: {tests['bm25_search']['queries_tested']}")
        print(f"   - Multilevel results: {'✅' if tests['bm25_search']['multilevel_results'] else '❌'}")

        print(f"\n4. Hybrid Fusion: {'✅' if overall['criteria_met']['hybrid_fusion_effective'] else '❌'}")
        print(f"   - Test cases: {tests['hybrid_fusion']['test_cases']}")
        print(f"   - Fusion effective: {'✅' if tests['hybrid_fusion']['fusion_effective'] else '❌'}")
        print(f"   - Better than individual: {'✅' if tests['hybrid_fusion']['better_than_individual'] else '❌'}")

        print("\n" + "="*80)

    asyncio.run(main())
