You are an agent working in the repo. First, read & index the codebase to fully understand current structure, modules, and paths. Then implement the following changes precisely. Don’t introduce new architectural layers; keep it lean and practical.

PHASE 0 — READ & ORIENT
- Open and read these files end-to-end:
  - src/services/crawler/web_scraper.py
  - src/services/crawler/metadata_extractor.py
  - src/services/crawler/file_downloader.py
  - src/services/crawler/crawler_orchestrator.py
  - src/services/pdf/extractor.py
  - src/services/pdf/ocr_extractor.py
  - src/services/pdf/pdf_orchestrator.py
  - src/services/ingestion.py
  - src/db/models.py
  - src/services/utils.py
- Note: pdf_orchestrator imports "build_citation_string" from ...utils.citation. If that path doesn’t exist, remove that import and inline a minimal citation builder or skip it (do not block pipeline).

PHASE 1 — DEFINE RAW_DOC CONTRACT
- Create a dataclass or Pydantic model for RAW_DOC in src/services/pdf/schemas_raw.py:
  - artifact_type: literal "RAW_DOC"
  - doc: fields (doc_id, doc_form, doc_number, doc_year, title, dates, status, language, source_urls[], pdf_info{pages, sha256?, ocr_lang?, ocr_conf_avg?}, relationships{}, flags{is_amendment, is_omnibus})
  - document_tree: root node {"type":"dokumen","unit_id":<doc_id>,"label_display":<title>,"children":[...]}
  - Node types supported: "preamble","menimbang","mengingat","bab","bagian","paragraf","pasal","ayat","huruf","angka","penjelasan","penjelasan_pasal","omnibus_container","butir".
  - For nodes, support: unit_id, label_display, text_raw, children[], spans[{page,int,y1?,y2?}], payload_pasal_blocks[] for omnibus-butir.
- Add a JSON schema generator if useful later.

PHASE 2 — METADATA NORMALIZATION
- In metadata_extractor.py, normalize extracted HTML data into a dict that maps exactly to RAW_DOC.doc fields:
  - doc_form, doc_number, doc_year, title
  - date_enacted/promulgated/effective as YYYY-MM-DD or null
  - status, language="id"
  - source_urls = [detail_url, pdf_url]
  - relationships = {mengubah[], diubah_dengan[], mencabut[], mencabut_sebagian[], uji_materi[]}
  - flags.is_amendment = heuristic: title contains "Perubahan atas" OR relationships.mengubah not empty; otherwise false.
- Do NOT attempt to parse pasal structure here.

PHASE 3 — PDF EXTRACTION (TEXT + PAGES)
- In pdf_orchestrator.py:
  - Expose a single public method:
    def build_raw_doc(document_meta: dict, pdf_path: str, enable_ocr_fallback: bool = True) -> dict:
      returns a RAW_DOC dict.
  - Use UnifiedPDFExtractor first; if low confidence (<60) and enable_ocr_fallback, try OCRExtractor.
  - Extract: text (full), page_count, confidence, method, processing_time.
  - Also extract per-page text using PyMuPDF (fitz) if available:
    - get per-page block text (or simple page-wise text) and keep a list [ {"page":1,"start_line":i,"end_line":j}, ... ] to map lines to pages.

PHASE 4 — TREE BUILDER (HIERARCHY + OMNIBUS)
- Replace/extend _build_document_tree to:
  - Preserve existing hierarchy detection for buku/bab/bagian/paragraf/pasal/ayat/huruf/angka.
  - Add OMNIBUS detection:
    - Detect “PASAL I/II/III …” headings: ^\s*PASAL\s+([IVXLC]+)\b
    - Inside each PASAL omnibus, scan for “payung”:
      Beberapa\s+ketentuan\s+dalam\s+Undang-Undang.+?diubah\s+sebagai\s+berikut:
      If found, set container.type="omnibus_container", attach omnibus_target if possible (parse UU target from the sentence if present).
    - Split numbered items (butir):
      (?m)^\s*(\d{1,3})\.\s+(?=(Ketentuan|Di antara|Diantara|Penjelasan|Di\s+antara)\b)
      For each butir, store:
        - number_label (e.g., "11")
        - preface_text_raw: the classifier trigger line(s)
        - block_text_raw: capture content after “berbunyi sebagai berikut:” until the next butir/PASAL/end.
        - payload_pasal_blocks[]: if multiple "Pasal X" blocks present within block_text_raw, split them with regex ^\s*Pasal\s+(\d+[A-Z]?)\s*$
        - spans: page only (mandatory), y1/y2 optional
    - Triggers (classification ONLY for metadata; we still just emit RAW):
      replace/insert/repeal/replace_explanation/phrase_edit/renumber-hint.
  - Minimal spans:
    - For each new node (pasal and butir), record the page number based on the first line index that created the node.

PHASE 5 — FLAGS & OUTPUT
- After building the tree:
  - Set flags.is_omnibus = true if any omnibus_container present.
- Merge document_meta + tree into RAW_DOC dict:
  - artifact_type="RAW_DOC"
  - doc.pdf_info.pages = page_count
  - doc.pdf_info.ocr_conf_avg = confidence if OCR used, else null or extractor’s score
  - doc.source_urls preserved
- Return the dict; DO NOT write files here.

PHASE 6 — INGESTION CLI
- In ingestion.py:
  - Add flags:
    --export-raw (default true)
    --raw-dir (default ./data/raw)
    --skip-ocr (default false)
  - After crawl & pdf download, for each document with pdf_path, call PDFOrchestrator.build_raw_doc(meta, pdf_path, enable_ocr_fallback=not args.skip_ocr) and save output as {raw_dir}/{doc_id}.json
  - Ensure directories exist.

PHASE 7 — TESTS (basic, no external PDF)
- Create tests that:
  1) Feed synthetic multi-line strings representing:
     - UU non-amendment with simple bab/pasal/ayat/huruf (expect normal tree).
     - UU omnibus with PASAL I and three butir: replace, insert (51A & 51B), repeal.
     - Replace penjelasan.
  2) Use a fake extractor that returns (text, page_count=3).
  3) Assert RAW_DOC shape:
     - artifact_type, doc fields
     - flags.is_amendment, flags.is_omnibus
     - presence of omnibus_container and its butir children
     - payload_pasal_blocks length for multi-pasal.
  4) Confirm spans.page is set for pasal and butir nodes.

IMPORTANT IMPLEMENTATION DETAILS
- unit_id conventions:
  - normal: "{doc_id}/bab-I/pasal-6/ayat-1/huruf-a" (use printed labels, not array indexes)
  - omnibus: "{doc_id}/pasal-I/angka-11" for butir #11
- Avoid brittle imports like "...utils.citation"; if missing, remove and proceed.
- Be lenient to OCR/noisy whitespace; always normalize line endings and collapse multiple spaces.
- Do not attempt consolidation or write INDEX_PACKAGE here.

ACCEPTANCE
- Running ingestion with --export-raw produces RAW_DOC JSON files that pass shape validation and include omnibus_container with butir for at least the synthetic omnibus fixture.
- Non-omnibus documents produce a clean hierarchical tree with pasal/ayat/huruf/angka nodes.
- Minimal page spans exist for pasal and butir nodes.