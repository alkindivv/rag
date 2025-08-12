You are an agent working inside my codebase. Your job: 
1) Read and index the repository files I provided (they may contain syntax errors). 
2) Fix only what is broken to make the pipeline runnable.
3) Implement an in-place upgrade so that the current crawler → downloader → extractor/OCR → pdf_orchestrator → ingestion pipeline outputs ONE enriched JSON per document that includes metadata AND a full document_tree (with omnibus/“butir” when present) and minimal page spans — WITHOUT adding new CLI args, new artifact types, or new orchestration layers.

====================================================
PHASE 0 — READ & REPAIR (NO ARCHITECTURE CHANGES)
====================================================
Open and read these files fully:
- models.py
- web_scraper.py
- metadata_extractor.py
- file_downloader.py
- extractor.py
- ocr_extractor.py
- pdf_orchestrator.py
- crawler_orchestrator.py
- ingestion.py
- utils.py

A. Fix syntax/runtime errors minimally (many files appear truncated). 
   - Do not refactor structure; keep existing function names and public surface intact as much as possible.
   - If an import like utils.citation.build_citation_string doesn’t exist, remove it and inline a minimal helper or skip it — DO NOT block the pipeline.

B. Run a local quick static check (lint/type if available). 
   - If tests exist, run them. (If not, add a tiny sanity test later.)

====================================================
PHASE 1 — IN-PLACE SCHEMA UPGRADE (METADATA)
====================================================
Goal: Keep returning ONE Python dict for each document (metadata), but enrich it IN-PLACE so it contains:
- doc_id, doc_form, doc_number, doc_year, title,
- date_enacted, date_promulgated, date_effective (YYYY-MM-DD or null),
- status, language="id",
- source_urls = [detail_url, pdf_url],
- relationships = {mengubah[], diubah_dengan[], mencabut[], mencabut_sebagian[], uji_materi[]},
- pdf_info = {pages:int, sha256?:str, ocr_lang?:[str], ocr_conf_avg?:float},
- flags = {is_amendment: bool, is_omnibus: bool}, 
- document_tree = {...}   <-- will be injected by pdf_orchestrator later.

Tasks:
1) In metadata_extractor.py, normalize the returned metadata dict to exactly these keys (do not create new artifact objects or CLI options). 
   - Derive flags.is_amendment = true if the title contains “Perubahan atas” or relationships.mengubah is non-empty; else false.
   - Set flags.is_omnibus = false (will be set by pdf_orchestrator if omnibus detected).

2) Keep web_scraper/crawler_orchestrator behavior intact; they should pass this dict along.

====================================================
PHASE 2 — PDF ORCHESTRATOR BUILDS TREE (IN-PLACE)
====================================================
Implement/upgrade a single public function in pdf_orchestrator.py:

def build_document_tree(meta: dict, pdf_path: str, extractor_hint: str | None = None) -> dict:
    """Reads the PDF, extracts text (UnifiedPDFExtractor; fallback to OCR if needed), 
    builds a hierarchical tree (bab→pasal→ayat/huruf/angka), detects omnibus PASAL I/II… with numbered 'butir', 
    assigns minimal page spans, sets flags.is_omnibus if detected, 
    fills meta['pdf_info'] (pages, ocr_conf_avg if OCR, sha256 if available),
    and returns the SAME meta dict with meta['document_tree'] populated."""

Requirements (NO new CLI args):
- Use the existing extractor first; if confidence <0.6 or text too sparse, fallback to OCR extractor (if available).
- Ensure we can map lines to pages. If extractor doesn’t return per-page text, use PyMuPDF (fitz) inside pdf_orchestrator to get page-wise text. 
  Minimal span is {"page": <int>}. y1/y2 are optional (add if easy).
- Build the normal hierarchy you already support (buku/bab/bagian/paragraf/pasal/ayat/huruf/angka).
- OMNIBUS detection:
  * Identify “PASAL I/II/III …” headings:  (?im)^\s*PASAL\s+([IVXLC]+)\b
  * Detect omnibus umbrella:  Beberapa\s+ketentuan\s+dalam\s+Undang-Undang.+?diubah\s+sebagai\s+berikut:
  * Split numbered items (butir) only at line starts:
      (?m)^\s*(\d{1,3})\.\s+(?=(Ketentuan|Di antara|Diantara|Penjelasan|Di\s+antara)\b)
  * For each butir, capture:
      - number_label (“11”, “1”, …),
      - preface_text_raw (trigger line(s), e.g., “Ketentuan Pasal 1 diubah ...”),
      - block_text_raw (everything after “… berbunyi sebagai berikut:” until next butir/PASAL/end),
      - payload_pasal_blocks[] when block_text_raw contains multiple “Pasal X” blocks:
         split by:  (?im)^\s*Pasal\s+(\d+[A-Z]?)\s*$
      - spans: at least {"page": page_no} for the butir start.
  * If any omnibus_container is present, set meta["flags"]["is_omnibus"] = True.

- For normal (non-omnibus) documents, just build the usual bab/pasal/ayat tree; meta["flags"]["is_omnibus"] stays False.

- Always return the SAME meta dict with:
  meta["document_tree"] = {...}
  meta["pdf_info"]["pages"] = <int>
  meta["pdf_info"]["ocr_conf_avg"] = <float or null>
  meta["pdf_info"]["sha256"] = <keep from downloader if available>

====================================================
PHASE 3 — DOWNLOADER & EXTRACTOR TWEAKS (MINIMAL)
====================================================
- file_downloader.py: if not already doing so, compute sha256 of the PDF and attach it to the meta["pdf_info"]["sha256"] (do not change function signatures; you can return a tuple or set a field on meta if the orchestrator passes meta into the downloader).
- extractor.py / ocr_extractor.py: do not change the CLI. If feasible, expose per-page text to pdf_orchestrator via an internal method; otherwise pdf_orchestrator uses PyMuPDF to get per-page lines.

====================================================
PHASE 4 — ORCHESTRATION (NO NEW ARGS)
====================================================
- crawler_orchestrator.py keeps the same entrypoints. After metadata and download:
    meta = metadata_extractor(...);
    pdf_path, sha256 = file_downloader(...);
    meta["pdf_info"]["sha256"] = sha256   # if not already set
    meta = pdf_orchestrator.build_document_tree(meta, pdf_path)
    return meta

- ingestion.py keeps writing ONE JSON per document (no new flags):
  Save the meta (now enriched with document_tree, flags, pdf_info) to the existing output path, filename "{doc_id}.json".

====================================================
PHASE 5 — REGEX & ROBUSTNESS DETAILS
====================================================
Use these robust patterns (make whitespace/OCR tolerant):
- read utils/pattern_manager.py for pattern
- omnibus PASAL header:   (?im)^\\s*PASAL\\s+([IVXLC]+)\\b
- omnibus umbrella:       Beberapa\\s+ketentuan\\s+dalam\\s+Undang-Undang.+?diubah\\s+sebagai\\s+berikut:
- numbered item (butir):  (?m)^\\s*(\\d{1,3})\\.\\s+(?=(Ketentuan|Di antara|Diantara|Penjelasan|Di\\s+antara)\\b)
- trigger “replace pasal”: Ketentuan\\s+Pasal\\s+(\\d+[A-Z]?)\\s+diubah.*?berbunyi\\s+sebagai\\s+berikut:
- trigger “insert pasal”:  Di\\s+antara\\s+Pasal\\s+(\\d+[A-Z]?)\\s+dan\\s+Pasal\\s+(\\d+[A-Z]?)\\s+d[ai]sisipkan.*?berbunyi\\s+sebagai\\s+berikut:
- trigger “repeal pasal”:  Ketentuan\\s+Pasal\\s+\\d+[A-Z]?\\s+dihapus\\.
- trigger penjelasan:      Penjelasan\\s+Pasal\\s+(\\d+[A-Z]?)(?:\\s+ayat\\s+\\((\\d+)\\))?(?:\\s+huruf\\s+([a-z]))?\\s+diubah.*?berbunyi\\s+sebagai\\s+berikut:
- phrase edit:             (?:frasa|kata)\\s+\\"([^\\"]+)\\"\\s+diubah\\s+menjadi\\s+\\"([^\\"]+)\\"
- split multi-Pasal in a block:  (?im)^\\s*Pasal\\s+(\\d+[A-Z]?)\\s*$

When computing page spans:
- Maintain a running mapping of (line_index → page_no). For each node created (pasal, butir), record the first line’s page as spans[0].page.

====================================================
PHASE 6 — ACCEPTANCE (MUST PASS)
====================================================
1) For a non-amendment PDF:
   - Output JSON contains: doc_* fields, relationships{}, pdf_info.pages, flags{is_amendment=false, is_omnibus=false}, and a document_tree with bab→pasal→(ayat/huruf/angka).
   - Each pasal node has a deterministic unit_id like "{doc_id}/bab-I/pasal-1".

2) For an omnibus amendment PDF:
   - Output JSON includes an omnibus_container under document_tree with label_display "PASAL I" (and others if present).
   - It has children “butir” nodes with number_label (“1”, “11”, ...), preface_text_raw, block_text_raw (when applicable), payload_pasal_blocks[] if multi-pasal, and spans with at least {"page": <int>}.
   - flags.is_amendment can remain as metadata-extractor decided; pdf_orchestrator MUST set flags.is_omnibus=true if omnibus_container exists.

3) Spans minimality:
   - At least pasal and butir nodes have spans[0].page set correctly.

4) No new CLI flags, no new top-level artifact types. 
   - The orchestrator returns the SAME metadata dict, enriched in-place with document_tree and pdf_info; ingestion writes ONE JSON per doc as before.

5) Keep existing module layout; only add small helpers within existing files if absolutely necessary (do not create a new “raw” layer).

6) Lint passes; if any tiny tests exist, keep them green. If none, add one smoke test (string-fed) to verify omnibus parsing (butir detection + payload_pasal_blocks split).

Deliverables:
- Patched files (only the ones listed), 
- remove all duplicateion, redundancy and duplicate method, function and method

- A short README section (or code comments) explaining the enriched metadata fields and where document_tree is attached,
- Confirmation that running the existing ingestion path produces one JSON per doc with the enriched structure.
