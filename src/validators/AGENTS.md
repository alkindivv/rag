# AGENT — JSON Validator & Data Sanity Checks
**Tujuan:** Mengimplementasikan validator ketat untuk JSON schema V2 dan data sanity checks sebelum ingestion

## Overview
Modul ini bertanggung jawab untuk memvalidasi struktur JSON dokumen hukum sebelum proses ingestion. Validator memastikan unit_id unik dan deterministik, parent pointers sesuai hierarchy, number_label sesuai tipe, dan semua field wajib ada.

## Scope & Boundaries
- In-scope:
  - JSON V2 schema validation
  - Unit ID uniqueness and determinism checks
  - Parent pointer validation
  - Number label type validation
  - Pre-ingestion data sanity checks
- Out-of-scope:
  - Document extraction
  - Database storage
  - Retrieval logic

## Inputs & Outputs
- Input utama:
  - JSON document structures
  - Schema definitions
- Output:
  - Validation results
  - Error reports for invalid documents
- Artefak file:
  - `src/validators/json_validator.py`
  - `src/validators/sanity_checker.py`

## Dependencies
- Membutuhkan anchor checklist:
  - data.json.schema (anchor:data.json.schema)
- Menyediakan anchor checklist:
  - data.json.validator (anchor:data.json.validator)

## [PLANNING]
1. Implementasi JSON V2 validator
2. Membangun unit ID uniqueness checker
3. Menambahkan parent pointer validation
4. Mengembangkan number label type validator
5. Membuat pre-ingestion sanity checks

## [EXECUTION]
1. Buat `src/validators/json_validator.py` dengan validator ketat
2. Implementasi checks untuk:
   - unit_id unik dan deterministik (berbasis path)
   - parent_pasal_id untuk ayat; parent_ayat_id untuk huruf; parent_huruf_id untuk angka
   - number_label: pasal/ayat/angka = digit (allow A/B, bis/ter), huruf = a-z
   - citation_string ada di node pasal
   - raw_text_path ada (file .txt)
   - amendments field sesuai spec
3. Jalankan `pytest -q tests/unit/test_json_validator.py`
4. Dokumentasi di `src/validators/README.md`

## [VERIFICATION]
- JSON validator menolak documents yang tidak sesuai schema
- Unit ID uniqueness checker mendeteksi duplicates
- Parent pointer validation memastikan hierarchy yang benar
- Number label validator memeriksa format yang sesuai
- Sanity checks mencegah ingestion data yang invalid

## [TESTS]
- Unit:
  - `tests/unit/test_json_validator.py` - schema validation
  - `tests/unit/test_unit_id_checker.py` - uniqueness and determinism
  - `tests/unit/test_parent_pointer_validator.py` - parent relationships
  - `tests/unit/test_number_label_validator.py` - label format validation
- E2E:
  - Pre-ingestion validation pipeline
  - Error reporting untuk documents yang invalid
- Cara jalan cepat: `python tests/run_tests.py --quick src/validators`

## Acceptance Criteria
- (1) Validator menolak JSON yang tidak sesuai schema V2
- (2) Unit ID harus unik dan deterministik berbasis path
- (3) Parent pointers harus sesuai dengan level hierarchy
- (4) Number labels harus sesuai tipe yang ditentukan
- (5) Semua field wajib divalidasi dengan jelas

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:data.json.validator
```

## Notes
- Validator harus dijalankan sebelum ingestion process
- Error messages harus jelas dan actionable
- Validasi harus mencakup semua nested structures
