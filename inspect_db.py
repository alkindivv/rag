#!/usr/bin/env python3
"""
Database inspection script for Legal RAG system.
Examines the structure and content of legal documents and units.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.db.session import get_db_session
from sqlalchemy import text

def inspect_documents():
    """Inspect legal documents in the database."""
    print("=" * 60)
    print("LEGAL DOCUMENTS INSPECTION")
    print("=" * 60)

    with get_db_session() as db:
        # Document count
        result = db.execute(text('SELECT COUNT(*) as count FROM legal_documents')).fetchone()
        print(f"Total documents: {result.count}")

        # Document details
        result = db.execute(text('''
            SELECT doc_form, doc_number, doc_year, doc_title, doc_status, id
            FROM legal_documents
            ORDER BY doc_year DESC, doc_number
        ''')).fetchall()

        print("\nDocument Details:")
        for row in result:
            title_short = row.doc_title[:80] + "..." if len(row.doc_title) > 80 else row.doc_title
            print(f"  {row.doc_form} {row.doc_number}/{row.doc_year}: {title_short}")
            print(f"    Status: {row.doc_status}, ID: {row.id}")

def inspect_units():
    """Inspect legal units by type."""
    print("\n" + "=" * 60)
    print("LEGAL UNITS INSPECTION")
    print("=" * 60)

    with get_db_session() as db:
        # Unit count by type
        result = db.execute(text('''
            SELECT unit_type, COUNT(*) as count
            FROM legal_units
            GROUP BY unit_type
            ORDER BY count DESC
        ''')).fetchall()

        print("Units by type:")
        for row in result:
            print(f"  {row.unit_type}: {row.count}")

        # Sample units of each type
        unit_types = ['BAB', 'PASAL', 'AYAT', 'HURUF', 'ANGKA']

        for unit_type in unit_types:
            print(f"\nSample {unit_type} units:")
            result = db.execute(text('''
                SELECT ld.doc_form, ld.doc_number, ld.doc_year,
                       lu.unit_type, lu.number_label, lu.citation_string,
                       lu.unit_id, lu.hierarchy_path
                FROM legal_units lu
                JOIN legal_documents ld ON ld.id = lu.document_id
                WHERE lu.unit_type = :unit_type
                ORDER BY ld.doc_year DESC, lu.number_label
                LIMIT 5
            '''), {'unit_type': unit_type}).fetchall()

            for row in result:
                print(f"  {row.doc_form} {row.doc_number}/{row.doc_year} {unit_type} {row.number_label}")
                print(f"    Citation: {row.citation_string}")
                print(f"    Unit ID: {row.unit_id}")
                print(f"    Hierarchy: {row.hierarchy_path}")

def test_citation_parsing():
    """Test citation parsing with real data."""
    print("\n" + "=" * 60)
    print("CITATION PARSING TEST WITH REAL DATA")
    print("=" * 60)

    from src.services.citation.parser import get_best_citation_match

    # Test with real documents
    test_queries = [
        "UU 24/2019 Pasal 1",
        "UU 1/2023 Pasal 5",
        "UU 24/2012 Pasal 10",
        "Pasal 1 UU 24/2019",
        "UU 24/2019",
        "Pasal 5 ayat (1)"
    ]

    for query in test_queries:
        print(f"\nTesting: '{query}'")
        citation_match = get_best_citation_match(query)
        if citation_match:
            print(f"  Parsed: {citation_match.to_dict()}")
        else:
            print("  No match found")

def test_database_queries():
    """Test database queries with real citations."""
    print("\n" + "=" * 60)
    print("DATABASE QUERY TEST WITH REAL DATA")
    print("=" * 60)

    from src.services.citation.parser import get_best_citation_match

    with get_db_session() as db:
        # Test with UU 24/2019 Pasal 1
        query = "UU 24/2019 Pasal 1"
        citation_match = get_best_citation_match(query)

        if citation_match:
            print(f"Testing database query for: {citation_match.to_dict()}")

            # Build similar query to what the system uses
            sql_query = '''
                SELECT lu.unit_id, lu.unit_type, lu.number_label, lu.citation_string,
                       ld.doc_form, ld.doc_number, ld.doc_year
                FROM legal_units lu
                JOIN legal_documents ld ON ld.id = lu.document_id
                WHERE ld.doc_form = :doc_form
                  AND ld.doc_number = :doc_number
                  AND ld.doc_year = :doc_year
                  AND (lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)
                ORDER BY lu.unit_type, lu.number_label
                LIMIT 10
            '''

            params = {
                'doc_form': citation_match.doc_form,
                'doc_number': citation_match.doc_number,
                'doc_year': citation_match.doc_year,
                'pasal_number': citation_match.pasal_number
            }

            result = db.execute(text(sql_query), params).fetchall()

            print(f"Query returned {len(result)} results:")
            for row in result:
                print(f"  {row.unit_type} {row.number_label}: {row.citation_string}")

def check_citation_display_issues():
    """Check for citation display issues."""
    print("\n" + "=" * 60)
    print("CITATION DISPLAY ISSUES CHECK")
    print("=" * 60)

    with get_db_session() as db:
        # Check for BAB units that might be mislabeled as Pasal
        result = db.execute(text('''
            SELECT unit_type, number_label, citation_string, unit_id
            FROM legal_units
            WHERE unit_type = 'BAB' AND citation_string LIKE '%Pasal%'
            LIMIT 10
        ''')).fetchall()

        if result:
            print("BAB units with 'Pasal' in citation string (POTENTIAL BUG):")
            for row in result:
                print(f"  {row.unit_type} {row.number_label}: {row.citation_string}")
        else:
            print("No BAB units with 'Pasal' in citation string found")

        # Check for PASAL units that might be mislabeled as BAB
        result = db.execute(text('''
            SELECT unit_type, number_label, citation_string, unit_id
            FROM legal_units
            WHERE unit_type = 'PASAL' AND citation_string LIKE '%BAB%'
            LIMIT 10
        ''')).fetchall()

        if result:
            print("\nPASAL units with 'BAB' in citation string (POTENTIAL BUG):")
            for row in result:
                print(f"  {row.unit_type} {row.number_label}: {row.citation_string}")
        else:
            print("\nNo PASAL units with 'BAB' in citation string found")

def check_hierarchy_relationships():
    """Check hierarchy relationships."""
    print("\n" + "=" * 60)
    print("HIERARCHY RELATIONSHIPS CHECK")
    print("=" * 60)

    with get_db_session() as db:
        # Check HURUF units and their parent relationships
        result = db.execute(text('''
            SELECT lu.unit_type, lu.number_label, lu.parent_pasal_id,
                   lu.parent_ayat_id, lu.hierarchy_path
            FROM legal_units lu
            WHERE lu.unit_type = 'HURUF'
            LIMIT 10
        ''')).fetchall()

        print("HURUF units and their parent relationships:")
        for row in result:
            print(f"  HURUF {row.number_label}")
            print(f"    Parent PASAL ID: {row.parent_pasal_id}")
            print(f"    Parent AYAT ID: {row.parent_ayat_id}")
            print(f"    Hierarchy: {row.hierarchy_path}")

        # Check if parent relationships are properly set
        result = db.execute(text('''
            SELECT COUNT(*) as count
            FROM legal_units
            WHERE unit_type = 'HURUF' AND parent_pasal_id IS NULL
        ''')).fetchone()

        if result.count > 0:
            print(f"\nWARNING: {result.count} HURUF units have NULL parent_pasal_id")

def main():
    """Main inspection function."""
    print("🔍 Legal RAG Database Inspection")
    print("This script examines the database content to identify potential issues.")

    try:
        inspect_documents()
        inspect_units()
        test_citation_parsing()
        test_database_queries()
        check_citation_display_issues()
        check_hierarchy_relationships()

        print("\n" + "=" * 60)
        print("✅ Inspection completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
