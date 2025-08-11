# AGENT — JSON Canonical Schema & Evolution
**Tujuan:** Mendefinisikan dan mendokumentasikan canonical JSON schema V2 untuk dokumen hukum Indonesia

## Overview
Modul ini bertanggung jawab untuk mendefinisikan dan mendokumentasikan struktur canonical JSON V2 untuk dokumen hukum Indonesia. Schema ini digunakan sebagai contract untuk semua components dalam sistem RAG, termasuk database, validators, dan answer builder.

## Scope & Boundaries
- In-scope:
  - JSON V2 schema definition
  - Schema documentation
  - Schema evolution tracking
  - Contoh struktur dokumen
- Out-of-scope:
  - Validator implementation
  - Database queries
  - Application logic

## Inputs & Outputs
- Input utama:
  - Requirements dari AGENTS.md
- Output:
  - Canonical JSON schema V2
  - Schema documentation
- Artefak file:
  - `src/schemas/json_v2.md`
  - `src/schemas/examples/`

## Dependencies
- Membutuhkan anchor checklist:
  - Tidak ada dependencies eksternal
- Menyediakan anchor checklist:
  - data.json.schema (anchor:data.json.schema)

## [PLANNING]
1. Mendefinisikan struktur JSON V2 berdasarkan requirements
2. Membuat dokumentasi schema yang komprehensif
3. Menyediakan contoh struktur dokumen
4. Menambahkan notes tentang schema evolution
5. Validasi konsistensi dengan database models

## [EXECUTION]
1. Buat `src/schemas/json_v2.md` dengan spesifikasi lengkap:
   - Struktur doc object
   - Struktur content object
   - Tree hierarchy dengan parent pointers
   - Amendments structure
2. Tambahkan contoh struktur dokumen yang valid
3. Dokumentasikan parent relationships:
   - parent_pasal_id untuk ayat
   - parent_ayat_id untuk huruf
   - parent_huruf_id untuk angka
4. Buat direktori `src/schemas/examples/` dengan contoh dokumen
5. Validasi konsistensi dengan `src/db/models.py`

## [VERIFICATION]
- JSON schema V2 mencakup semua field yang diperlukan
- Parent relationships sesuai dengan requirements
- Amendments structure mendukung semua operation types
- Contoh dokumen valid sesuai dengan schema
- Konsistensi dengan database models terjaga

## [TESTS]
- Unit:
  - Tidak ada unit tests untuk schema definition
- E2E:
  - Schema validation against example documents
- Cara jalan cepat:
  - Manual review schema documentation

## Acceptance Criteria
- (1) JSON schema V2 terdokumentasi dengan jelas di `src/schemas/json_v2.md`
- (2) Schema mencakup semua field dan structures dari requirements
- (3) Parent pointers sesuai dengan level hierarchy yang ditentukan
- (4) Amendments structure mendukung semua operation types
- (5) Contoh dokumen valid tersedia untuk reference

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:data.json.schema
```

## Notes
- Schema harus mencerminkan struktur database yang ada
- Versioning harus diimplementasikan untuk tracking evolusi
- Contoh dokumen harus mencakup edge cases
