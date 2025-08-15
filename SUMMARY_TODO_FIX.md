UnitType.ANGKA_AMANDEMENT: di 
models.py
 ada enum ini, tapi orchestrator tidak pernah memproduksi type="angka_amandement". Putuskan salah satu:
Hilangkan enum itu dari indexer/mod





# Refactor Plan: Orchestrator-Only Aggregation + Simplified Hierarchy (parent_unit_id + ltree)

## Objective
- Make `src/pipeline/indexer.py` a pure consumer of orchestrator output (no aggregation/formatting logic).
- Keep all content assembly and marker formatting in `src/services/pdf/pdf_orchestrator.py` only.
- Replace specialized parent anchors (`parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id`) with:
  - `parent_unit_id` (immediate parent)
  - `unit_path` (Postgres ltree) for fast ancestor/descendant queries.
- Remove unused `UnitType.ANGKA_AMANDEMENT` and any logic that references it.

## Scope
- No shared text util module will be introduced. The orchestrator is the single source of truth for aggregation and formatting.
- Indexer consumes fields produced by orchestrator: `content`, `local_content`, `display_text`, `bm25_body`, `label_display`, `path`, `citation_string`, `parent_unit_id`, `unit_path`.

## Changes by Component

### 1) Database Schema (Alembic Migrations)
- Add columns to `legal_units`:
  - `parent_unit_id TEXT` with index.
  - `unit_path LTREE` with GIST (or GIN) index. Ensure `CREATE EXTENSION ltree`.
- Backfill `unit_path` from existing `unit_id`/`path` using a deterministic converter:
  - Lowercase; replace `-` and spaces with `_`; strip non-alnum; join with dots, e.g.: `uu_2011_24.bab_1.pasal_5.ayat_2.huruf_b`.
- Backfill `parent_unit_id` using the immediate parent `unit_id` derived from `path`.
- Update all read paths (services/queries) to use `unit_path` operations:
  - Descendants of a node: `WHERE unit_path <@ '...pasal_5'`.
  - Ancestors of a node: `WHERE unit_path @> '...ayat_2'` (if needed).
- Deprecate and later drop columns:
  - `parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id`.

### 2) Orchestrator (`src/services/pdf/pdf_orchestrator.py`)
- Ensure fallback cleaning works: in `_clean_text()`, call `self._basic_clean(text)` when `TextCleaner` unavailable.
- Keep and finalize aggregation logic here only:
  - `_build_full_content_for_pasal()` remains and must assemble complete pasal content (children included) and correct markers (angka uses `"1."`).
  - Ensure all PASAL units have non-empty `content` (already patched per SUMMARY_TODO_FIX).
- During `_serialize_tree()`:
  - Set `parent_unit_id` (immediate parent `unit_id`).
  - Build and set `unit_path` (ltree-friendly) for every node.
  - Populate `content/local_content/display_text/bm25_body/label_display/path/citation_string`.

### 3) Indexer (`src/pipeline/indexer.py`)
- Remove duplicated aggregation/formatting:
  - Delete/disable `_build_pasal_aggregated_content()` and `_build_full_content()`.
- Persist exactly what orchestrator emits:
  - `content`, `local_content`, `display_text`, `bm25_body`.
  - `parent_unit_id`, `unit_path`.
- Embeddings:
  - Use `unit.content` for PASAL units directly (no rebuild).
- Clean imports and remove any references to `UnitType.ANGKA_AMANDEMENT`.

### 4) Models (`src/db/models.py`)
- Add mapped columns:
  - `parent_unit_id = Column(Text, index=True)`
  - `unit_path = Column(LtreeType or custom ltree, index=True)`
- Remove enum `ANGKA_AMANDEMENT` from `UnitType`.
- Keep existing indexes for vectors and doc metadata; add ltree indexes for `legal_units.unit_path`.

### 5) Query Layer / Services
- Replace filters based on `parent_*` with `unit_path` predicates.
  - Example: all ayat under pasal X → `unit_path <@ :pasal_path`.
- Where needed, still use `parent_unit_id` for direct parent hops.

## Rollout Plan
1. Migration 1: Add `parent_unit_id`, `unit_path`, create indexes, enable `ltree`.
2. Code change: Orchestrator populates `parent_unit_id` + `unit_path`. Indexer writes them.
3. Backfill task: compute `unit_path` and `parent_unit_id` for existing rows.
4. Switch all queries to `unit_path` (and/or `parent_unit_id`).
5. Migration 2: Drop `parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id` after verification.

## Validation & Tests
- Unit tests:
  - Orchestrator builds correct `unit_path` and `parent_unit_id` for mixed hierarchies.
  - Every PASAL has non-empty `content` and correct `angka` markers ("1.").
- Integration tests:
  - Query descendants using `unit_path <@` vs previous logic → same result set.
  - Indexer persists orchestrator fields without re-aggregation; embeddings match `content`.
- E2E:
  - Hybrid search returns comparable/better results. No regression in latency.

## Risks & Mitigations
- ltree availability: ensure `CREATE EXTENSION ltree` in provisioning; document fallback (text + trigram) if absolutely necessary.
- Backfill correctness: write deterministic converter; add audit query to find rows with invalid `unit_path`.
- Query changes: wrap `unit_path` predicates in helper functions to avoid repeated string building.

## Tasks Checklist
- [ ] Alembic migration: add `parent_unit_id`, `unit_path`, indexes, enable `ltree`.
- [ ] Orchestrator: populate `parent_unit_id` and `unit_path`; ensure content aggregation finalized.
- [ ] Indexer: remove aggregation functions; consume orchestrator fields; cleanup imports.
- [ ] Models: add new columns/types; remove `ANGKA_AMANDEMENT`.
- [ ] Backfill script for historical data.
- [ ] Update query services to use `unit_path`.
- [ ] Remove old parent_* columns (follow-up migration).
- [ ] Tests: unit, integration, E2E updated and passing.

---

## Audit: Citation Parser and Ltree Integration

### Findings (Current State)
- __Parser location__: `src/services/citation/parser.py` with `CitationMatch` and `LegalCitationParser`.
- __Scope__: Regex-based extraction for UU/PP; supports pasal/ayat/huruf/angka; no ltree path emitted.
- __Gaps vs ltree plan__:
  - __No unit_path output__: Parser does not construct ltree-compatible `unit_path` or `unit_path_prefix`.
  - __Roman numerals__: Sections like `Bab I/II`, `Bagian A/B` not parsed/mapped.
  - __Pasal suffix__: Variants `14A`, `27B`, `bis/ter/quater` not normalized.
  - __Ranges__: `ayat (1)-(3)` or `huruf a–c` not supported.
  - __Cross-reference__: Phrases `sebagaimana dimaksud pada Pasal ...` not detected.
  - __Doc alias resolution__: `UU Minerba`, `UU Cipta Kerja` → number/year mapping absent.
  - __No span metadata__: Start/end offsets for highlighting not captured.
  - __Confidence routing__: OK, but not integrated with ltree output.
  - __Duplication risk__: `src/utils/citation.py` builds display strings; scope says avoid shared text utils—prefer orchestrator as SoT.

### Design Decision
- __Keep parser, make it thin__: Parser focuses on extraction + normalization; traversal and hierarchy are handled by Postgres `ltree`.
- __Single source of truth for aggregation/formatting__: Keep in `src/services/pdf/pdf_orchestrator.py` as already planned.

### Minimal Interface (Proposed)
- Function: `parse_citation(text: str) -> List[CitationMatch]`
- Extend `CitationMatch` with ltree-ready fields:
  - `unit_path_exact: Optional[str]`  e.g., `uu_2009_4.pasal_149.ayat_2.huruf_b`
  - `unit_path_prefix: Optional[str]` e.g., `uu_2009_4.pasal_149`
  - `span: Optional[Tuple[int,int]]` (optional; useful for highlighting)
- Normalized components (lowercase, underscore, alnum-only) consistent with unit_path converter in this plan.

### Normalization Rules (Authoritative)
- Document node: `uu_{year}_{number}` (or `pp_{year}_{number}`, etc.)
- Pasal: `pasal_{num}` where `num` may include suffix e.g., `14a`, `27b`, `14bis` → map to `14bis`.
- Ayat: `ayat_{n}` (digits only)
- Huruf: `huruf_{a..z}`
- Angka (sublist): `angka_{n}`
- Roman sections if present:
  - `bab_{roman_to_int}` (e.g., `Bab IV` → `bab_4`),
  - `bagian_{roman_to_int}` when applicable.

### Query Usage (ltree)
- Exact node: `WHERE unit_path = :unit_path_exact`
- Subtree: `WHERE unit_path <@ :unit_path_prefix`
- Parent/child hops: `parent_unit_id` when a direct parent is needed; otherwise prefer ltree.

### Concrete Actions
- __A1. Extend `CitationMatch`__ in `src/services/citation/parser.py` with `unit_path_exact`, `unit_path_prefix`, optional `span`.
- __A2. Build unit_path__ inside parser using deterministic converter (same rules as backfill): lowercase, replace `[-\s]`→`_`, strip non-alnum except `_`, join with dots.
- __A3. Add pattern support__:
  - Roman numerals for `Bab/Bagian` (non-blocking; optional field in path).
  - Suffix pasal (`A/B`, `bis/ter/quater`).
  - Ranges for ayat/huruf (expand to multiple matches or mark prefix at parent).
  - Cross-ref phrasing detection to bump confidence for explicit path.
- __A4. Doc alias map (optional)__: lightweight in-memory alias→(form, number, year) for common acts (Minerba, Cipta Kerja). Pluggable via config.
- __A5. Emit ltree strings only__: no DB traversal in parser. Return both `unit_path_exact` (if pasal/ayat/huruf complete) and `unit_path_prefix` (if only pasal/doc level).
- __A6. Update consumers__:
  - `VectorSearchService._lookup_citation_units`: switch to `unit_path` predicates from emitted fields.
  - `QueryOptimizationService`: use `is_explicit_citation()` as-is; no change in contract.
- __A7. Deprecate duplicate display utils__:
  - Prefer orchestrator for display strings; if `src/utils/citation.py` is used for rendering only, move or reference from orchestrator to avoid divergence.

### Test Plan Additions
- Parser unit tests: parsing matrix for UU/PP, suffixes, ranges, roman sections, cross-ref phrasings.
- Integration: explicit queries resolve to exact nodes via `unit_path` equality; subtree queries via prefix produce children (huruf/angka) correctly.

### Checklist Addendum (Parser)
- [ ] `parser.py`: add `unit_path_exact`, `unit_path_prefix`, `span` to `CitationMatch`.
- [ ] `parser.py`: implement deterministic unit_path builder.
- [ ] `parser.py`: add patterns for roman, suffix, ranges, cross-ref.
- [ ] `vector_search.py`: use `unit_path` filters from parser outputs.
- [ ] Optional: alias mapping for popular act nicknames.
- [ ] Review/remove duplicate `src/utils/citation.py` usage; centralize in orchestrator if needed.
