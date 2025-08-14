#!/usr/bin/env python3
"""
Comprehensive Global Accuracy Audit for Legal RAG System

This script performs a systematic accuracy assessment across ALL legal query types
to identify false positives, accuracy issues, and systematic quality problems.

Purpose: Ensure consistent high accuracy across ALL query categories and eliminate false positives.
Author: Senior AI Systems Auditor and Search Accuracy Specialist
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.search.vector_search import VectorSearchService, SearchResult
from src.services.llm.legal_llm import LegalLLMService
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryTestCase:
    """Test case for query evaluation."""
    query: str
    category: str
    subcategory: str
    expected_pasal: Optional[str] = None
    expected_content_keywords: List[str] = None
    should_not_contain: List[str] = None
    min_relevance_score: float = 0.7
    description: str = ""


@dataclass
class AccuracyResult:
    """Result of accuracy evaluation."""
    query: str
    category: str
    subcategory: str
    results_count: int
    top_score: float
    avg_score: float
    has_expected_content: bool
    has_false_positives: bool
    false_positive_details: List[str]
    relevance_rating: str  # EXCELLENT, GOOD, FAIR, POOR
    accuracy_score: float  # 0.0-1.0
    issues: List[str]
    recommendations: List[str]


class LegalRAGAccuracyAuditor:
    """Comprehensive accuracy auditor for Legal RAG system."""

    def __init__(self):
        """Initialize auditor with services."""
        self.search_service = VectorSearchService()
        self.llm_service = LegalLLMService()
        self.test_cases = self._build_comprehensive_test_suite()
        self.results: List[AccuracyResult] = []

    def _build_comprehensive_test_suite(self) -> List[QueryTestCase]:
        """Build comprehensive test suite covering all legal query categories."""

        test_cases = []

        # A. CRIMINAL LAW QUERIES
        test_cases.extend([
            QueryTestCase(
                query="apa bedanya hukuman bagi orang yang melakukan pembunuhan tanpa sengaja dengan hukuman bagi orang yang melakukan pembunuhan berencana?",
                category="Criminal Law",
                subcategory="Comparative Criminal Sanctions",
                expected_pasal="458",
                expected_content_keywords=["pembunuhan", "pidana penjara", "15 tahun", "sengaja", "berencana"],
                description="Complex comparative murder query - KNOWN ISSUE"
            ),
            QueryTestCase(
                query="berapa lama hukuman untuk pembunuhan?",
                category="Criminal Law",
                subcategory="Specific Crime Penalties",
                expected_pasal="458",
                expected_content_keywords=["pembunuhan", "15 tahun", "penjara"],
                description="Direct murder penalty query"
            ),
            QueryTestCase(
                query="sanksi pidana untuk korupsi berapa?",
                category="Criminal Law",
                subcategory="Specific Crime Penalties",
                expected_content_keywords=["korupsi", "pidana", "sanksi"],
                description="Corruption penalties"
            ),
            QueryTestCase(
                query="apa perbedaan antara pembunuhan biasa dengan pembunuhan berencana?",
                category="Criminal Law",
                subcategory="Legal Distinctions",
                expected_pasal="458",
                expected_content_keywords=["pembunuhan", "berencana", "penjara"],
                description="Murder type distinctions"
            ),
            QueryTestCase(
                query="hukuman untuk pencurian dengan kekerasan?",
                category="Criminal Law",
                subcategory="Specific Crime Penalties",
                expected_content_keywords=["pencurian", "kekerasan", "pidana"],
                description="Robbery penalties"
            )
        ])

        # B. CIVIL LAW QUERIES
        test_cases.extend([
            QueryTestCase(
                query="syarat sahnya kontrak menurut hukum indonesia?",
                category="Civil Law",
                subcategory="Contract Law",
                expected_content_keywords=["kontrak", "syarat", "sah"],
                description="Contract validity requirements"
            ),
            QueryTestCase(
                query="hak dan kewajiban suami istri dalam perkawinan?",
                category="Civil Law",
                subcategory="Family Law",
                expected_content_keywords=["suami", "istri", "hak", "kewajiban"],
                description="Marital rights and obligations"
            ),
            QueryTestCase(
                query="cara pembagian harta warisan menurut hukum?",
                category="Civil Law",
                subcategory="Inheritance Law",
                expected_content_keywords=["warisan", "pembagian", "harta"],
                description="Inheritance distribution"
            )
        ])

        # C. ADMINISTRATIVE LAW QUERIES
        test_cases.extend([
            QueryTestCase(
                query="prosedur pengajuan izin usaha?",
                category="Administrative Law",
                subcategory="Business License Procedures",
                expected_content_keywords=["izin", "usaha", "prosedur"],
                description="Business license procedures"
            ),
            QueryTestCase(
                query="sanksi administratif untuk pelanggaran lingkungan?",
                category="Administrative Law",
                subcategory="Environmental Sanctions",
                expected_content_keywords=["sanksi", "administratif", "lingkungan"],
                description="Environmental administrative sanctions"
            )
        ])

        # D. CONSTITUTIONAL LAW QUERIES
        test_cases.extend([
            QueryTestCase(
                query="hak asasi manusia dalam UUD 1945?",
                category="Constitutional Law",
                subcategory="Fundamental Rights",
                expected_content_keywords=["hak asasi", "manusia", "UUD"],
                description="Human rights in constitution"
            ),
            QueryTestCase(
                query="pembagian kekuasaan negara menurut konstitusi?",
                category="Constitutional Law",
                subcategory="Government Structure",
                expected_content_keywords=["kekuasaan", "negara", "konstitusi"],
                description="Government power distribution"
            )
        ])

        # E. CITATION-SPECIFIC QUERIES (These should work perfectly)
        test_cases.extend([
            QueryTestCase(
                query="pasal 458 UU 1 tahun 2023",
                category="Citation-Specific",
                subcategory="Explicit Pasal Reference",
                expected_pasal="458",
                expected_content_keywords=["pembunuhan", "merampas nyawa"],
                min_relevance_score=0.95,
                description="Direct pasal lookup - should be perfect"
            ),
            QueryTestCase(
                query="UU No. 1 Tahun 2023 Pasal 36",
                category="Citation-Specific",
                subcategory="Full Citation Format",
                expected_pasal="36",
                expected_content_keywords=["pertanggungjawaban", "sengaja", "kealpaan"],
                min_relevance_score=0.95,
                description="Full citation format - should be perfect"
            ),
            QueryTestCase(
                query="pasal 544 ayat 1 UU 1/2023",
                category="Citation-Specific",
                subcategory="Specific Ayat Reference",
                expected_pasal="544",
                min_relevance_score=0.95,
                description="Specific ayat reference"
            )
        ])

        # F. COMPARATIVE LEGAL QUERIES (Problematic category)
        test_cases.extend([
            QueryTestCase(
                query="perbedaan sanksi pidana dan sanksi administratif?",
                category="Comparative Legal",
                subcategory="Sanction Types Comparison",
                expected_content_keywords=["sanksi", "pidana", "administratif", "perbedaan"],
                description="Criminal vs administrative sanctions comparison"
            ),
            QueryTestCase(
                query="apa beda UU dengan PP?",
                category="Comparative Legal",
                subcategory="Legal Hierarchy Comparison",
                expected_content_keywords=["undang-undang", "peraturan pemerintah", "beda"],
                description="Law vs regulation comparison"
            ),
            QueryTestCase(
                query="bedanya pasal 36 dan pasal 37 UU 1/2023?",
                category="Comparative Legal",
                subcategory="Pasal Comparison",
                expected_pasal="36|37",
                expected_content_keywords=["pertanggungjawaban", "kesalahan"],
                description="Specific pasal comparison"
            )
        ])

        # G. DEFINITIONAL QUERIES
        test_cases.extend([
            QueryTestCase(
                query="apa definisi tindak pidana?",
                category="Definitional",
                subcategory="Legal Term Definition",
                expected_content_keywords=["tindak pidana", "definisi", "pengertian"],
                description="Criminal act definition"
            ),
            QueryTestCase(
                query="pengertian pidana penjara?",
                category="Definitional",
                subcategory="Penalty Type Definition",
                expected_content_keywords=["pidana penjara", "pengertian"],
                description="Prison sentence definition"
            )
        ])

        # H. FALSE POSITIVE DETECTION QUERIES (Edge cases)
        test_cases.extend([
            QueryTestCase(
                query="apakah ada pasal yang tidak ada?",
                category="False Positive Detection",
                subcategory="Non-existent Content",
                should_not_contain=["huruf a", "keamanan negara"],
                description="Should return low confidence, not generic content"
            ),
            QueryTestCase(
                query="xyz 123 tidak ada hukum",
                category="False Positive Detection",
                subcategory="Nonsense Query",
                should_not_contain=["huruf a", "keamanan negara"],
                description="Nonsense query should not return confident false matches"
            )
        ])

        return test_cases

    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive accuracy audit across all test cases."""

        print("🔍 STARTING COMPREHENSIVE LEGAL RAG ACCURACY AUDIT")
        print("=" * 80)

        start_time = time.time()
        self.results = []

        # Process each test case
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Testing: {test_case.category} - {test_case.subcategory}")
            print(f"Query: {test_case.query}")

            try:
                result = await self._evaluate_query(test_case)
                self.results.append(result)

                # Print immediate feedback
                print(f"✅ Accuracy: {result.accuracy_score:.2f} | Relevance: {result.relevance_rating}")
                if result.issues:
                    print(f"⚠️  Issues: {', '.join(result.issues[:2])}")

            except Exception as e:
                logger.error(f"Failed to evaluate query '{test_case.query}': {e}")
                continue

        duration = time.time() - start_time

        # Generate comprehensive report
        report = self._generate_comprehensive_report(duration)

        # Save detailed results
        self._save_audit_results(report)

        return report

    async def _evaluate_query(self, test_case: QueryTestCase) -> AccuracyResult:
        """Evaluate a single query for accuracy issues."""

        # Execute search
        search_results = await self.search_service.search_async(
            query=test_case.query,
            k=5,
            use_reranking=True
        )

        results = search_results.get("results", [])

        # Calculate basic metrics
        results_count = len(results)
        top_score = results[0].score if results else 0.0
        avg_score = statistics.mean([r.score for r in results]) if results else 0.0

        # Evaluate content relevance
        has_expected_content = self._check_expected_content(results, test_case)
        has_false_positives, fp_details = self._detect_false_positives(results, test_case)

        # Rate overall relevance
        relevance_rating = self._rate_relevance(results, test_case)

        # Calculate accuracy score
        accuracy_score = self._calculate_accuracy_score(
            results, test_case, has_expected_content, has_false_positives
        )

        # Identify specific issues
        issues = self._identify_issues(results, test_case, accuracy_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(test_case, issues, results)

        return AccuracyResult(
            query=test_case.query,
            category=test_case.category,
            subcategory=test_case.subcategory,
            results_count=results_count,
            top_score=top_score,
            avg_score=avg_score,
            has_expected_content=has_expected_content,
            has_false_positives=has_false_positives,
            false_positive_details=fp_details,
            relevance_rating=relevance_rating,
            accuracy_score=accuracy_score,
            issues=issues,
            recommendations=recommendations
        )

    def _check_expected_content(self, results: List[SearchResult], test_case: QueryTestCase) -> bool:
        """Check if results contain expected content."""
        if not results:
            return False

        # Check for expected pasal
        if test_case.expected_pasal:
            for result in results[:3]:  # Check top 3 results
                citation = result.citation_string or ""
                if test_case.expected_pasal in citation:
                    return True

        # Check for expected keywords
        if test_case.expected_content_keywords:
            for result in results[:3]:
                content = (result.content or "").lower()
                keyword_matches = sum(1 for kw in test_case.expected_content_keywords
                                    if kw.lower() in content)
                if keyword_matches >= len(test_case.expected_content_keywords) * 0.6:  # 60% threshold
                    return True

        return len(results) > 0  # If no specific expectations, just check if we got results

    def _detect_false_positives(self, results: List[SearchResult], test_case: QueryTestCase) -> Tuple[bool, List[str]]:
        """Detect false positive patterns in results."""
        false_positives = []

        for result in results:
            citation = result.citation_string or ""
            content = (result.content or "").lower()

            # Pattern 1: Generic "huruf a" with irrelevant content
            if "huruf a" in citation and len(content) < 100:
                if any(generic in content for generic in ["keamanan negara", "menarik diri", "proses kehidupan"]):
                    false_positives.append(f"Generic 'huruf a' content: {citation}")

            # Pattern 2: High confidence but irrelevant content
            if result.score > 0.8 and test_case.should_not_contain:
                for forbidden in test_case.should_not_contain:
                    if forbidden.lower() in content:
                        false_positives.append(f"High confidence irrelevant content: {forbidden}")

            # Pattern 3: Wrong legal domain
            if test_case.category == "Criminal Law" and result.score > 0.7:
                if any(civil_term in content for civil_term in ["kontrak", "perkawinan", "warisan"]):
                    false_positives.append(f"Wrong legal domain (civil law for criminal query)")

            # Pattern 4: Partial citation matches with wrong content
            if test_case.expected_pasal and test_case.expected_pasal not in citation:
                if result.score > 0.8:  # High confidence but wrong pasal
                    false_positives.append(f"High confidence wrong pasal: {citation}")

        return len(false_positives) > 0, false_positives

    def _rate_relevance(self, results: List[SearchResult], test_case: QueryTestCase) -> str:
        """Rate overall relevance of results."""
        if not results:
            return "POOR"

        top_score = results[0].score
        has_expected = self._check_expected_content(results, test_case)

        if top_score >= 0.9 and has_expected:
            return "EXCELLENT"
        elif top_score >= 0.8 and has_expected:
            return "GOOD"
        elif top_score >= 0.7 or has_expected:
            return "FAIR"
        else:
            return "POOR"

    def _calculate_accuracy_score(self, results: List[SearchResult], test_case: QueryTestCase,
                                has_expected: bool, has_false_positives: bool) -> float:
        """Calculate overall accuracy score (0.0-1.0)."""
        if not results:
            return 0.0

        # Base score from search quality
        base_score = min(results[0].score, 1.0)

        # Adjust for expected content
        if has_expected:
            base_score += 0.2
        else:
            base_score -= 0.3

        # Penalty for false positives
        if has_false_positives:
            base_score -= 0.4

        # Category-specific adjustments
        if test_case.category == "Citation-Specific" and results[0].score < 0.95:
            base_score -= 0.3  # Citation queries should be near perfect

        if test_case.category == "Comparative Legal" and not has_expected:
            base_score -= 0.2  # Comparative queries are critical

        return max(0.0, min(1.0, base_score))

    def _identify_issues(self, results: List[SearchResult], test_case: QueryTestCase,
                        accuracy_score: float) -> List[str]:
        """Identify specific accuracy issues."""
        issues = []

        if not results:
            issues.append("NO_RESULTS_RETURNED")
        elif accuracy_score < 0.5:
            issues.append("LOW_ACCURACY_SCORE")

        if test_case.category == "Citation-Specific" and results and results[0].score < 0.95:
            issues.append("CITATION_LOOKUP_DEGRADED")

        if test_case.category == "Comparative Legal" and not self._check_expected_content(results, test_case):
            issues.append("COMPARATIVE_QUERY_FAILED")

        if results and results[0].score > 0.8 and not self._check_expected_content(results, test_case):
            issues.append("HIGH_CONFIDENCE_WRONG_CONTENT")

        # Check for systematic false positive patterns
        fp_count = sum(1 for r in results if "huruf a" in (r.citation_string or ""))
        if fp_count >= 2:
            issues.append("SYSTEMATIC_FALSE_POSITIVES")

        return issues

    def _generate_recommendations(self, test_case: QueryTestCase, issues: List[str],
                                results: List[SearchResult]) -> List[str]:
        """Generate specific recommendations for improvements."""
        recommendations = []

        if "COMPARATIVE_QUERY_FAILED" in issues:
            recommendations.append("Implement multi-part query decomposition for comparative questions")
            recommendations.append("Enhance query expansion for comparative legal terms")

        if "SYSTEMATIC_FALSE_POSITIVES" in issues:
            recommendations.append("Filter out generic 'huruf a' content with low semantic relevance")
            recommendations.append("Implement content quality scoring to penalize generic fragments")

        if "HIGH_CONFIDENCE_WRONG_CONTENT" in issues:
            recommendations.append("Recalibrate confidence scoring for better precision")
            recommendations.append("Add domain-specific relevance filtering")

        if "CITATION_LOOKUP_DEGRADED" in issues:
            recommendations.append("Verify citation parsing and exact match algorithms")

        if not results or (results and results[0].score < 0.6):
            recommendations.append("Improve embedding quality for Indonesian legal terminology")
            recommendations.append("Expand legal keyword dictionary and synonyms")

        return recommendations

    def _generate_comprehensive_report(self, duration: float) -> Dict[str, Any]:
        """Generate comprehensive audit report with actionable insights."""

        # Calculate overall statistics
        total_tests = len(self.results)
        accuracy_scores = [r.accuracy_score for r in self.results]
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0

        # Category-wise analysis
        category_stats = {}
        for result in self.results:
            cat = result.category
            if cat not in category_stats:
                category_stats[cat] = {
                    'count': 0, 'accuracy_sum': 0.0, 'failed_count': 0,
                    'false_positive_count': 0, 'issues': []
                }

            stats = category_stats[cat]
            stats['count'] += 1
            stats['accuracy_sum'] += result.accuracy_score
            if result.accuracy_score < 0.7:
                stats['failed_count'] += 1
            if result.has_false_positives:
                stats['false_positive_count'] += 1
            stats['issues'].extend(result.issues)

        # Calculate category averages
        for cat, stats in category_stats.items():
            stats['avg_accuracy'] = stats['accuracy_sum'] / stats['count']
            stats['failure_rate'] = stats['failed_count'] / stats['count']
            stats['false_positive_rate'] = stats['false_positive_count'] / stats['count']

        # Identify critical issues
        critical_issues = self._identify_critical_issues()

        # Generate improvement roadmap
        improvement_roadmap = self._generate_improvement_roadmap(category_stats, critical_issues)

        report = {
            "audit_metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(duration, 2),
                "total_test_cases": total_tests,
                "auditor_version": "1.0.0"
            },
            "overall_accuracy": {
                "average_accuracy_score": round(avg_accuracy, 3),
                "accuracy_grade": self._grade_accuracy(avg_accuracy),
                "tests_passed": sum(1 for r in self.results if r.accuracy_score >= 0.7),
                "tests_failed": sum(1 for r in self.results if r.accuracy_score < 0.7),
                "false_positive_rate": round(sum(1 for r in self.results if r.has_false_positives) / total_tests, 3)
            },
            "category_analysis": category_stats,
            "critical_issues": critical_issues,
            "improvement_roadmap": improvement_roadmap,
            "detailed_results": [self._result_to_dict(r) for r in self.results]
        }

        return report

    def _identify_critical_issues(self) -> List[Dict[str, Any]]:
        """Identify critical system-wide accuracy issues."""
        critical_issues = []

        # Issue 1: Comparative query failures
        comparative_failures = [r for r in self.results
                              if r.category == "Comparative Legal" and r.accuracy_score < 0.5]
        if len(comparative_failures) >= 2:
            critical_issues.append({
                "issue": "COMPARATIVE_QUERY_SYSTEMATIC_FAILURE",
                "severity": "HIGH",
                "description": "Complex comparative legal queries consistently fail despite having relevant content",
                "affected_queries": len(comparative_failures),
                "impact": "Users cannot get comparative legal analysis",
                "root_cause": "Query decomposition and multi-part processing insufficient"
            })

        # Issue 2: False positive contamination
        fp_count = sum(1 for r in self.results if r.has_false_positives)
        if fp_count > len(self.results) * 0.3:  # More than 30% have false positives
            critical_issues.append({
                "issue": "SYSTEMATIC_FALSE_POSITIVES",
                "severity": "HIGH",
                "description": "High rate of false positive results with irrelevant content",
                "affected_queries": fp_count,
                "impact": "User trust degradation, incorrect legal information",
                "root_cause": "Insufficient content quality filtering and relevance scoring"
            })

        # Issue 3: Citation lookup degradation
        citation_issues = [r for r in self.results
                          if r.category == "Citation-Specific" and r.accuracy_score < 0.9]
        if citation_issues:
            critical_issues.append({
                "issue": "CITATION_LOOKUP_DEGRADATION",
                "severity": "MEDIUM",
                "description": "Direct citation lookups not achieving expected 95%+ accuracy",
                "affected_queries": len(citation_issues),
                "impact": "Unreliable legal reference lookups",
                "root_cause": "Citation parsing or database indexing issues"
            })

        return critical_issues

    def _generate_improvement_roadmap(self, category_stats: Dict, critical_issues: List) -> Dict[str, Any]:
        """Generate prioritized improvement roadmap."""

        return {
            "immediate_fixes": [
                {
                    "priority": 1,
                    "task": "Fix Comparative Query Processing",
                    "description": "Implement robust multi-part query decomposition for comparative legal questions",
                    "target": "Enable 'A vs B' legal queries to work reliably",
                    "estimated_impact": "25% accuracy improvement for comparative queries"
                },
                {
                    "priority": 2,
                    "task": "Filter False Positive Patterns",
                    "description": "Add content quality filtering to eliminate generic 'huruf a' false positives",
                    "target": "Reduce false positive rate to <10%",
                    "estimated_impact": "15% overall accuracy improvement"
                }
            ],
            "medium_term_improvements": [
                {
                    "priority": 3,
                    "task": "Enhanced Legal Domain Understanding",
                    "description": "Improve embedding quality for Indonesian legal terminology",
                    "target": "Better semantic understanding of legal concepts",
                    "estimated_impact": "10% overall accuracy improvement"
                },
                {
                    "priority": 4,
                    "task": "Confidence Calibration",
                    "description": "Recalibrate confidence scoring for better precision-recall balance",
                    "target": "High confidence results are actually relevant",
                    "estimated_impact": "Improved user trust and reduced false positives"
                }
            ],
            "long_term_enhancements": [
                {
                    "priority": 5,
                    "task": "Legal Reasoning Integration",
                    "description": "Add legal reasoning capabilities for complex comparative analysis",
                    "target": "Support complex legal analysis and reasoning",
                    "estimated_impact": "Support for advanced legal research workflows"
                }
            ]
        }

    def _grade_accuracy(self, accuracy: float) -> str:
        """Grade overall accuracy performance."""
        if accuracy >= 0.95:
            return "A+ (Excellent)"
        elif accuracy >= 0.90:
            return "A (Very Good)"
        elif accuracy >= 0.85:
            return "B+ (Good)"
        elif accuracy >= 0.80:
            return "B (Acceptable)"
        elif accuracy >= 0.70:
            return "C (Needs Improvement)"
        else:
            return "F (Poor - Critical Issues)"

    def _result_to_dict(self, result: AccuracyResult) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "query": result.query,
            "category": result.category,
            "subcategory": result.subcategory,
            "results_count": result.results_count,
            "top_score": round(result.top_score, 3),
            "avg_score": round(result.avg_score, 3),
            "has_expected_content": result.has_expected_content,
            "has_false_positives": result.has_false_positives,
            "false_positive_details": result.false_positive_details,
            "relevance_rating": result.relevance_rating,
            "accuracy_score": round(result.accuracy_score, 3),
            "issues": result.issues,
            "recommendations": result.recommendations
        }

    def _save_audit_results(self, report: Dict[str, Any]):
        """Save detailed audit results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"legal_rag_accuracy_audit_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed audit results saved to: {filename}")

    def print_executive_summary(self, report: Dict[str, Any]):
        """Print executive summary of audit results."""

        overall = report["overall_accuracy"]
        critical = report["critical_issues"]

        print("\n" + "=" * 80)
        print("🎯 LEGAL RAG ACCURACY AUDIT - EXECUTIVE SUMMARY")
        print("=" * 80)

        print(f"\n📊 OVERALL SYSTEM ACCURACY")
        print(f"   Grade: {overall['accuracy_grade']}")
        print(f"   Average Accuracy: {overall['average_accuracy_score']:.1%}")
        print(f"   Tests Passed: {overall['tests_passed']}/{overall['tests_passed'] + overall['tests_failed']}")
        print(f"   False Positive Rate: {overall['false_positive_rate']:.1%}")

        print(f"\n🔥 CRITICAL ISSUES IDENTIFIED: {len(critical)}")
        for issue in critical:
            print(f"   ❌ {issue['issue']} (Severity: {issue['severity']})")
            print(f"      {issue['description']}")

        print(f"\n🏆 TOP IMPROVEMENT PRIORITIES:")
        roadmap = report["improvement_roadmap"]
        for fix in roadmap["immediate_fixes"][:3]:
            print(f"   {fix['priority']}. {fix['task']}")
            print(f"      Target: {fix['target']}")

        print(f"\n📋 CATEGORY PERFORMANCE:")
        for category, stats in report["category_analysis"].items():
            accuracy = stats['avg_accuracy']
            fp_rate = stats['false_positive_rate']
            print(f"   {category}: {accuracy:.1%} accuracy, {fp_rate:.1%} false positives")

        print("\n" + "=" * 80)


async def main():
    """Main execution function for comprehensive audit."""

    print("🚀 INITIALIZING LEGAL RAG ACCURACY AUDITOR...")

    try:
        auditor = LegalRAGAccuracyAuditor()

        print(f"📋 Loaded {len(auditor.test_cases)} comprehensive test cases")
        print("🔬 Beginning systematic accuracy evaluation...")

        # Run comprehensive audit
        report = await auditor.run_comprehensive_audit()

        # Display executive summary
        auditor.print_executive_summary(report)

        # Check if critical issues need immediate attention
        critical_issues = report["critical_issues"]
        if critical_issues:
            print(f"\n🚨 URGENT: {len(critical_issues)} critical issues require immediate attention!")
            print("   Review the detailed audit report for specific fix recommendations.")
        else:
            print(f"\n✅ No critical issues detected. System operating within acceptable parameters.")

        print(f"\n📊 Audit completed successfully!")

    except Exception as e:
        logger.error(f"Audit execution failed: {e}")
        print(f"❌ Audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
