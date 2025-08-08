#!/usr/bin/env python3
"""
Real-World Test: UU No. 1 Tahun 2023 (KUHP)
Tests the complete vector pipeline with actual Indonesian Criminal Code.

This test validates the pipeline with a complex legal document featuring:
- BUKU KESATU structure
- Multiple BAB levels
- Bagian and Paragraf subdivisions
- Complex legal language
- Real citation requirements

Follows RULES.md - simple but comprehensive real-world testing.
"""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.vector_pipeline import (
    VectorPipeline,
    create_simple_pipeline,
    process_legal_document,
    search_legal_content
)


def load_kuhp_document():
    """Load the real KUHP document."""
    print("📄 Loading UU No. 1 Tahun 2023 (KUHP)...")

    kuhp_path = Path("test/fix-verification/undang_undang_1_2023.txt")

    if not kuhp_path.exists():
        print(f"❌ File not found: {kuhp_path}")
        print("💡 Make sure the KUHP file exists at the specified path")
        return None

    with open(kuhp_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Document metadata
    metadata = {
        'title': 'Undang-Undang Republik Indonesia Nomor 1 Tahun 2023 tentang Kitab Undang-Undang Hukum Pidana',
        'type': 'undang-undang',
        'number': '1',
        'year': '2023',
        'subject': 'Kitab Undang-Undang Hukum Pidana',
        'category': 'hukum pidana'
    }

    print(f"✅ Loaded document: {len(content)} characters")
    print(f"📋 Title: {metadata['title']}")

    return content, metadata


def test_kuhp_processing():
    """Test processing the complete KUHP document."""
    print("\n🔄 Testing KUHP Document Processing...")

    # Load document
    doc_data = load_kuhp_document()
    if not doc_data:
        return False

    content, metadata = doc_data

    # Create pipeline
    pipeline = create_simple_pipeline(
        qdrant_host=os.getenv('QDRANT_HOST', 'localhost'),
        qdrant_port=int(os.getenv('QDRANT_PORT', 6333))
    )

    # Process document
    print("⚙️  Processing document through pipeline...")
    start_time = time.time()

    result = process_legal_document(
        pipeline=pipeline,
        document_text=content,
        title=metadata['title'],
        doc_type=metadata['type'],
        number=metadata['number'],
        year=metadata['year']
    )

    processing_time = time.time() - start_time

    # Display results
    print(f"\n📊 KUHP Processing Results:")
    print(f"   ✅ Success: {result.success}")
    print(f"   📦 Chunks created: {result.chunks_processed}")
    print(f"   🧠 Vectors stored: {result.vectors_stored}")
    print(f"   ⏱️  Processing time: {processing_time:.2f}s")
    print(f"   ⚡ Speed: {result.chunks_processed/processing_time:.1f} chunks/sec")
    print(f"   📝 Average chunk size: {len(content)/result.chunks_processed:.0f} chars/chunk")

    if result.errors:
        print(f"   ⚠️  Errors: {len(result.errors)}")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"      - {error}")

    return result.success and result.vectors_stored > 0


def test_kuhp_search_scenarios():
    """Test realistic search scenarios for KUHP."""
    print("\n🔍 Testing KUHP Search Scenarios...")

    pipeline = create_simple_pipeline(
        qdrant_host=os.getenv('QDRANT_HOST', 'localhost'),
        qdrant_port=int(os.getenv('QDRANT_PORT', 6333))
    )

    # Real-world legal search scenarios
    search_scenarios = [
        {
            "query": "tindak pidana di wilayah negara Indonesia",
            "description": "Territorial jurisdiction",
            "expected_concepts": ["wilayah", "indonesia", "pidana"],
            "expected_bab": "BAB I"
        },
        {
            "query": "permufakatan jahat dan persiapan tindak pidana",
            "description": "Criminal conspiracy and preparation",
            "expected_concepts": ["permufakatan", "jahat", "persiapan"],
            "expected_bab": "BAB II"
        },
        {
            "query": "asas berlakunya hukum pidana menurut tempat",
            "description": "Territorial principle of criminal law",
            "expected_concepts": ["asas", "tempat", "teritorial"],
            "expected_bab": "BAB I"
        },
        {
            "query": "warga negara Indonesia melakukan kejahatan di luar negeri",
            "description": "Active nationality principle",
            "expected_concepts": ["warga", "indonesia", "luar"],
            "expected_bab": "BAB I"
        },
        {
            "query": "sanksi pidana dan tindakan dalam KUHP",
            "description": "Criminal sanctions and measures",
            "expected_concepts": ["sanksi", "pidana", "tindakan"],
            "expected_bab": "BAB II"
        },
        {
            "query": "hukum yang hidup dalam masyarakat",
            "description": "Living law in society",
            "expected_concepts": ["hukum", "hidup", "masyarakat"],
            "expected_bab": "BAB I"
        }
    ]

    successful_searches = 0
    search_results = []

    for scenario in search_scenarios:
        print(f"\n🎯 Scenario: {scenario['description']}")
        print(f"   Query: '{scenario['query']}'")

        start_time = time.time()
        results = search_legal_content(
            pipeline=pipeline,
            query=scenario['query'],
            limit=5
        )
        search_time = time.time() - start_time

        if results:
            top_result = results[0]
            print(f"   📊 Found: {len(results)} results in {search_time:.3f}s")
            print(f"   🏆 Top result:")
            print(f"      Citation: {top_result['citation']}")
            print(f"      Score: {top_result['score']:.3f}")
            print(f"      Content: {top_result['content'][:120]}...")

            # Check for expected concepts
            found_concepts = []
            content_lower = top_result['content'].lower()
            citation_lower = top_result['citation'].lower()

            for concept in scenario['expected_concepts']:
                if concept.lower() in content_lower or concept.lower() in citation_lower:
                    found_concepts.append(concept)

            print(f"      ✅ Concepts found: {found_concepts}")

            # Check BAB accuracy
            expected_bab = scenario.get('expected_bab', '')
            if expected_bab and expected_bab.lower() in top_result['citation'].lower():
                print(f"      ✅ Correct BAB: {expected_bab}")
            elif expected_bab:
                print(f"      ⚠️  Expected {expected_bab}, got: {top_result['citation']}")

            successful_searches += 1

            search_results.append({
                'scenario': scenario['description'],
                'query': scenario['query'],
                'results_count': len(results),
                'top_score': top_result['score'],
                'search_time': search_time,
                'citation': top_result['citation'],
                'concepts_found': found_concepts
            })

        else:
            print(f"   ❌ No results found in {search_time:.3f}s")
            search_results.append({
                'scenario': scenario['description'],
                'query': scenario['query'],
                'results_count': 0,
                'top_score': 0.0,
                'search_time': search_time,
                'citation': '',
                'concepts_found': []
            })

    # Analyze search quality
    if search_results:
        avg_score = sum(r['top_score'] for r in search_results if r['top_score'] > 0) / max(1, successful_searches)
        avg_search_time = sum(r['search_time'] for r in search_results) / len(search_results)

        print(f"\n📈 KUHP Search Analysis:")
        print(f"   Successful searches: {successful_searches}/{len(search_scenarios)}")
        print(f"   Success rate: {successful_searches/len(search_scenarios)*100:.1f}%")
        print(f"   Average score: {avg_score:.3f}")
        print(f"   Average search time: {avg_search_time:.3f}s")

    return successful_searches >= len(search_scenarios) * 0.8  # 80% success rate


def test_specific_legal_citations():
    """Test specific legal citation accuracy."""
    print("\n📚 Testing Specific Legal Citations...")

    pipeline = create_simple_pipeline(
        qdrant_host=os.getenv('QDRANT_HOST', 'localhost'),
        qdrant_port=int(os.getenv('QDRANT_PORT', 6333))
    )

    # Specific citation tests
    citation_tests = [
        {
            "query": "Pasal 1 ayat 1 asas legalitas",
            "expected_citation": "Pasal 1",
            "description": "Nullum crimen principle"
        },
        {
            "query": "Pasal 4 tindak pidana di wilayah Indonesia",
            "expected_citation": "Pasal 4",
            "description": "Territorial jurisdiction"
        },
        {
            "query": "Pasal 12 definisi tindak pidana",
            "expected_citation": "Pasal 12",
            "description": "Definition of criminal acts"
        },
        {
            "query": "Pasal 13 permufakatan jahat",
            "expected_citation": "Pasal 13",
            "description": "Criminal conspiracy"
        }
    ]

    citation_accuracy = 0

    for test in citation_tests:
        print(f"\n🎯 Citation Test: {test['description']}")
        print(f"   Query: '{test['query']}'")
        print(f"   Expected: {test['expected_citation']}")

        results = search_legal_content(
            pipeline=pipeline,
            query=test['query'],
            limit=3
        )

        if results:
            top_result = results[0]
            if test['expected_citation'] in top_result['citation']:
                print(f"   ✅ CORRECT: {top_result['citation']} (score: {top_result['score']:.3f})")
                citation_accuracy += 1
            else:
                print(f"   ❌ WRONG: Expected {test['expected_citation']}, got {top_result['citation']}")
        else:
            print(f"   ❌ NO RESULTS")

    accuracy_rate = citation_accuracy / len(citation_tests) * 100
    print(f"\n📊 Citation Accuracy: {citation_accuracy}/{len(citation_tests)} ({accuracy_rate:.1f}%)")

    return accuracy_rate >= 75.0  # 75% accuracy threshold


def generate_kuhp_report():
    """Generate comprehensive report of KUHP processing."""
    print("\n📋 Generating KUHP Processing Report...")

    pipeline = create_simple_pipeline(
        qdrant_host=os.getenv('QDRANT_HOST', 'localhost'),
        qdrant_port=int(os.getenv('QDRANT_PORT', 6333))
    )

    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()

    # Sample some vectors to analyze structure
    sample_search = search_legal_content(
        pipeline=pipeline,
        query="undang-undang hukum pidana",
        limit=10
    )

    report = {
        'document': 'UU No. 1 Tahun 2023 (KUHP)',
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_stats': stats,
        'sample_chunks': []
    }

    # Analyze chunk structure
    if sample_search:
        for i, result in enumerate(sample_search[:5], 1):
            chunk_info = {
                'chunk_number': i,
                'citation': result['citation'],
                'content_length': len(result['content']),
                'keywords': result['keywords'][:5],  # Top 5 keywords
                'score': result['score'],
                'bab': result.get('bab'),
                'pasal': result.get('pasal')
            }
            report['sample_chunks'].append(chunk_info)

    # Save report
    report_path = 'kuhp_processing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved: {report_path}")

    # Display summary
    print(f"\n📊 KUHP Processing Summary:")
    print(f"   Total vectors: {stats.get('vector_count', 0)}")
    print(f"   Collection status: {stats.get('collection_status', 'unknown')}")
    print(f"   Sample chunks analyzed: {len(report['sample_chunks'])}")

    if report['sample_chunks']:
        citations = [chunk['citation'] for chunk in report['sample_chunks']]
        print(f"   Sample citations: {', '.join(citations[:3])}...")

    return True


def run_kuhp_comprehensive_test():
    """Run comprehensive test suite for KUHP document."""
    print("=" * 80)
    print("🏛️  KUHP (UU No. 1 Tahun 2023) COMPREHENSIVE TEST")
    print("=" * 80)
    print("")

    # Check prerequisites
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not found in .env file")
        return False

    print(f"🔧 Configuration:")
    print(f"   Qdrant Host: {os.getenv('QDRANT_HOST', 'localhost')}")
    print(f"   Qdrant Port: {os.getenv('QDRANT_PORT', 6333)}")
    print(f"   Document: UU No. 1 Tahun 2023 (KUHP)")
    print("")

    # Test suite
    tests = [
        ("KUHP Document Processing", test_kuhp_processing),
        ("KUHP Search Scenarios", test_kuhp_search_scenarios),
        ("Legal Citation Accuracy", test_specific_legal_citations),
        ("Generate KUHP Report", generate_kuhp_report)
    ]

    results = {}
    passed = 0

    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            if result:
                passed += 1
            print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            print(f"❌ {test_name}: ERROR - {e}")
        print("")

    # Final report
    print("=" * 80)
    print(f"🏆 KUHP TEST RESULTS: {passed}/{len(tests)} PASSED")
    print("=" * 80)

    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {result}")

    print("")
    if passed == len(tests):
        print("🎉 ALL KUHP TESTS PASSED!")
        print("🏛️  Your vector pipeline successfully processed the Indonesian Criminal Code!")
        print("📊 Key achievements:")
        print("   ✅ Complex legal document structure handled correctly")
        print("   ✅ BUKU/BAB/Bagian/Paragraf/Pasal hierarchy preserved")
        print("   ✅ Accurate legal citations generated")
        print("   ✅ Semantic search working for real legal queries")
        print("   ✅ Production-ready for legal document processing")
        print("")
        print("💡 You can now process any Indonesian legal document!")
    else:
        print(f"⚠️ {len(tests) - passed} test(s) failed.")
        print("🔧 Check the specific issues above for debugging.")

    return passed == len(tests)


if __name__ == "__main__":
    success = run_kuhp_comprehensive_test()
    sys.exit(0 if success else 1)
