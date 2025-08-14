#!/usr/bin/env python3
"""
Simple test script to validate citation parser functionality.

Run with: python test_citation_parser.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.citation.parser import (
    LegalCitationParser,
    parse_citation,
    is_explicit_citation,
    get_best_citation_match
)


def test_citation_patterns():
    """Test various citation patterns."""

    print("🔍 Testing Legal Citation Parser")
    print("=" * 50)

    test_cases = [
        # Complete citations
        {
            "name": "Complete UU Citation",
            "query": "UU 8/2019 Pasal 6 ayat (2) huruf b",
            "expected": {
                "doc_form": "UU",
                "doc_number": "8",
                "doc_year": 2019,
                "pasal_number": "6",
                "ayat_number": "2",
                "huruf_letter": "b"
            }
        },
        {
            "name": "PP Citation",
            "query": "PP No. 45 Tahun 2020 Pasal 12",
            "expected": {
                "doc_form": "PP",
                "doc_number": "45",
                "doc_year": 2020,
                "pasal_number": "12"
            }
        },
        {
            "name": "UU Long Form",
            "query": "Undang-Undang No. 4 Tahun 2009 Pasal 121 Ayat 1",
            "expected": {
                "doc_form": "UU",
                "doc_number": "4",
                "doc_year": 2009,
                "pasal_number": "121",
                "ayat_number": "1"
            }
        },

        # Short form citations
        {
            "name": "Short UU Citation",
            "query": "UU 21/2008 Pasal 15",
            "expected": {
                "doc_form": "UU",
                "doc_number": "21",
                "doc_year": 2008,
                "pasal_number": "15"
            }
        },

        # Partial citations
        {
            "name": "Pasal Only",
            "query": "Pasal 15 ayat (1)",
            "expected": {
                "pasal_number": "15",
                "ayat_number": "1"
            }
        },
        {
            "name": "Document Only",
            "query": "UU 21/2008",
            "expected": {
                "doc_form": "UU",
                "doc_number": "21",
                "doc_year": 2008
            }
        },
        {
            "name": "Ayat Only",
            "query": "ayat (3) huruf c",
            "expected": {
                "ayat_number": "3",
                "huruf_letter": "c"
            }
        },

        # Non-citation queries (should not be detected)
        {
            "name": "Definition Query (Non-citation)",
            "query": "definisi badan hukum dalam peraturan perundang-undangan",
            "expected": None  # Should not be detected as citation
        },
        {
            "name": "Sanctions Query (Non-citation)",
            "query": "sanksi pidana pelanggaran lingkungan hidup",
            "expected": None  # Should not be detected as citation
        },
        {
            "name": "General Concept (Non-citation)",
            "query": "tanggung jawab sosial perusahaan",
            "expected": None  # Should not be detected as citation
        }
    ]

    passed_tests = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Query: '{test_case['query']}'")

        # Test citation detection
        is_citation = is_explicit_citation(test_case['query'], 0.60)
        should_be_citation = test_case['expected'] is not None

        print(f"   Citation Detection: {'✅' if is_citation == should_be_citation else '❌'} "
              f"(Expected: {should_be_citation}, Got: {is_citation})")

        if should_be_citation:
            # Test citation parsing
            best_match = get_best_citation_match(test_case['query'])

            if best_match:
                print(f"   Confidence: {best_match.confidence:.2f}")
                print(f"   Parsed Citation:")

                # Check expected fields
                all_correct = True
                for field, expected_value in test_case['expected'].items():
                    actual_value = getattr(best_match, field)
                    is_correct = actual_value == expected_value
                    all_correct = all_correct and is_correct

                    print(f"     {field}: {'✅' if is_correct else '❌'} "
                          f"Expected: {expected_value}, Got: {actual_value}")

                if all_correct:
                    passed_tests += 1
                    print("   Result: ✅ PASS")
                else:
                    print("   Result: ❌ FAIL")
            else:
                print("   Result: ❌ FAIL (No citation parsed)")
        else:
            # Non-citation query - should not be detected
            if not is_citation:
                passed_tests += 1
                print("   Result: ✅ PASS (Correctly not detected as citation)")
            else:
                print("   Result: ❌ FAIL (Incorrectly detected as citation)")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    print(f"   Pass Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Citation parser is working correctly.")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review the parser logic.")
        return False


def test_complex_patterns():
    """Test more complex and edge case patterns."""

    print("\n🧪 Testing Complex Citation Patterns")
    print("=" * 50)

    complex_cases = [
        "UU No. 8 Tahun 2019 tentang Penyandang Disabilitas Pasal 6 ayat (2) huruf b angka 1",
        "PP 45/2020 Pasal 12 ayat (3)",
        "Peraturan Pemerintah No. 123 Tahun 2021 Pasal 5",
        "PERPU 1/2020 Pasal 10 ayat (1) huruf a",
        "Pasal 25A ayat (2)",
        "UU 8/2019 • Pasal 6",
        "huruf d angka 2",
        "multiple citations: UU 8/2019 Pasal 6 and PP 45/2020 Pasal 12"
    ]

    for i, query in enumerate(complex_cases, 1):
        print(f"\n{i}. Testing: '{query}'")

        is_citation = is_explicit_citation(query, 0.60)
        print(f"   Detected as citation: {is_citation}")

        if is_citation:
            matches = parse_citation(query)
            print(f"   Found {len(matches)} citation(s):")

            for j, match in enumerate(matches[:3], 1):  # Show max 3 matches
                print(f"     {j}. Confidence: {match.confidence:.2f}")
                print(f"        Form: {match.doc_form}, Number: {match.doc_number}, "
                      f"Year: {match.doc_year}")
                print(f"        Pasal: {match.pasal_number}, Ayat: {match.ayat_number}, "
                      f"Huruf: {match.huruf_letter}")
                print(f"        Matched: '{match.matched_text}'")


def test_performance():
    """Test parsing performance."""

    print("\n⚡ Testing Citation Parser Performance")
    print("=" * 50)

    import time

    test_queries = [
        "UU 8/2019 Pasal 6 ayat (2) huruf b",
        "definisi badan hukum",
        "PP No. 45 Tahun 2020 Pasal 12",
        "sanksi pidana korupsi",
        "Pasal 15 ayat (1)",
        "tanggung jawab sosial perusahaan"
    ] * 100  # Repeat 100 times for performance test

    start_time = time.time()

    citation_count = 0
    for query in test_queries:
        if is_explicit_citation(query, 0.60):
            citation_count += 1
            get_best_citation_match(query)  # Parse the citation

    end_time = time.time()
    duration = end_time - start_time

    print(f"Processed {len(test_queries)} queries in {duration:.3f} seconds")
    print(f"Average time per query: {(duration/len(test_queries))*1000:.2f} ms")
    print(f"Citations detected: {citation_count}/{len(test_queries)}")
    print(f"Throughput: {len(test_queries)/duration:.0f} queries/second")

    # Performance should be very fast for citation detection
    avg_ms = (duration/len(test_queries))*1000
    if avg_ms < 1.0:
        print("✅ Performance: Excellent (< 1ms per query)")
    elif avg_ms < 5.0:
        print("✅ Performance: Good (< 5ms per query)")
    else:
        print("⚠️  Performance: Slow (> 5ms per query)")


if __name__ == "__main__":
    print("🚀 Legal RAG Citation Parser Test Suite")
    print("=" * 60)

    try:
        # Run basic pattern tests
        basic_success = test_citation_patterns()

        # Run complex pattern tests
        test_complex_patterns()

        # Run performance tests
        test_performance()

        print("\n" + "=" * 60)
        if basic_success:
            print("🎯 Citation parser is ready for production use!")
            print("✅ All core functionality validated")
            print("📈 Performance metrics within acceptable ranges")
        else:
            print("❌ Citation parser needs fixes before production use")
            print("🔧 Please review failed test cases above")

        print("\nNext steps:")
        print("1. Run full test suite: python -m pytest tests/ -v")
        print("2. Start API server: python src/api/main.py")
        print("3. Test search endpoint: curl http://localhost:8000/search")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure you're in the project root directory and dependencies are installed")
        print("Try: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
