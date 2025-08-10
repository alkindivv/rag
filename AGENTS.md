
PROJECT: Legal RAG – Indonesia (Repo: https://github.com/alkindivv/rag, branch: cox)

ROLE
You are an expert code refactorer + retrieval engineer for Indonesian legal text. Your task is to deeply refactor and harden the entire query path (extract → normalize → index → retrieve → rerank → prompt → answer), ensuring constitutional-grade accuracy while keeping the system simple, fast, and maintainable. You may refactor files, move logic, and adjust schemas if (and only if) it reduces complexity and increases precision. Do not add microservices or heavy infra.

DELETE ALL UNUSED FILES AFTER REFACTOR

============================================================
0) MANIFESTO AKURASI KONSTITUSIONAL (DO NOT VIOLATE)
============================================================
"Setiap jawaban harus menelusuri hierarki konstitusional: 
Naskah Asli → Penjelasan Umum → Penjelasan Pasal demi Pasal → Perubahan Pertama → Perubahan Kedua → Perubahan Ketiga → Perubahan Keempat (dst) → Putusan MK → SE/Edaran Pelaksanaan."

— Kutip teks normatif secara LITERAL dari unit yang direferensikan.
— Jika konteks tidak cukup, jawab "Tidak cukup konteks dari kutipan" dan sarankan pasal/aturan terkait.
— Selalu kembalikan JSON machine-first berisi kutipan, sitasi, dan jejak penalaran + human rendering sesudahnya.
— Jangan halusinasi. Jangan memodifikasi redaksi pasal.

============================================================
1) JENIS DOKUMEN & REGEX IDENTITAS (PAKAI PERSIS)
============================================================
Tabel ringkas (jenis, singkatan, regex identifier, contoh, sumber):
- UUD 1945        | UUD 1945 | ^UUD 1945$                                   | "Pasal 33 UUD 1945"              | Setneg
- Undang-Undang   | UU       | ^(UU|Undang-Undang)\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$ | "UU 11/2020"                 | DPR/Setneg
- Perubahan UU    | UU       | ^Perubahan\s+(Pertama|Kedua|Ketiga|Keempat|Kelima)\s+atas\s+UU\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$ | "Perubahan Ketiga UU 23/2014" | DPR/Setneg
- Perpu           | PERPU    | ^(Perpu|Peraturan Pemerintah Pengganti UU)\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$ | "Perpu 1/2020" | Presiden
- PP              | PP       | ^(PP|Peraturan Pemerintah)\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$               | "PP 23/2018"   | Setneg
- Perpres         | PERPRES  | ^(Perpres|Peraturan Presiden)\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$            | "Perpres 10/2021" | Setneg
- Permen          | PERMEN   | ^(Permen|Peraturan Menteri)\s+([A-Z]{2,10})\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$ | "Permen PANRB 1/2022" | K/L
- POJK            | POJK     | ^POJK\s*(No|Nomor)?\s*(\d+)\/POJK\.(\d{2})\/(\d{4})$                        | "POJK 12/POJK.03/2023" | OJK
- SEOJK           | SEOJK    | ^SEOJK\s*(No|Nomor)?\s*(\d+)\/SEOJK\.(\d{2})\/(\d{4})$                      | "SEOJK 3/SEOJK.07/2022" | OJK
- SEMA            | SEMA     | ^SEMA\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$                                      | "SEMA 3/2023"  | MA
- SEMK            | SEMK     | ^SEMK\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$                                      | "SEMK 1/2023"  | MK
- Perda           | PERDA    | ^(Perda|Peraturan Daerah)\s*(Prov|Kab|Kota)?\s*(No|Nomor)?\s*(\d+)[/ ](\d{4})$ | "Perda DKI 3/2021" | Pemda

Catatan hierarki konten: Buku (opsional, KUHAP/KUHP), Bab (romawi), Bagian, Paragraf, Pasal, Ayat, Huruf (a–z), Angka (1), Penjelasan (Umum & Pasal demi Pasal), Lampiran.

============================================================
2) CANONICAL JSON V2 (SESUAI DB) + VALIDATOR
============================================================
Buat: 
- schemas/json.md (spesifikasi)
- validators/json_validator.py (validator ketat; jalankan sebelum ingest)

Struktur dokumen (contoh singkat, isi lengkap di md):
{
  "schema_version": "2.0",
  "doc": {
    "id_external": "UU-2024-2",
    "form": "UU",
    "number": "2",
    "year": 2024,
    "title": "...",
    "status": "Berlaku",
    "source": "BPK|DPR|Setneg|OJK|MA|MK|Pemda",
    "detail_url": "...",
    "pdf_url": "...",
    "uji_materi_pdf_url": "...",
    "dates": { "enacted": "YYYY-MM-DD", "promulgated": "...", "effective": "..." },
    "teu": "Indonesia, Pemerintah Pusat",
    "place_enacted": "Jakarta",
    "language": "id",
    "field": "HUKUM ...",
    "subjects": ["..."],
       "relationships": [
      // all parsed from crawler metadata (doc-level + unit-level) 
      { "type": "type is auto from crawler",        "target_doc_id": "UU-2011-12", "description": "..." },
      { "type": "type is auto from crawler",   "target_doc_id": "UU-2024-2",  "description": "..." },
      { "type": "type is auto from crawler",        "target_doc_id": "UU-2007-29", "description": "..." },
      { "type": "type is auto from crawler",         "target_unit_id": "UU-2007-29/pasal-5", "description": "..." }
    ],
    "uji_materi": [{
      "decision_number": "xx/PUU-XX/20xx",
      "pdf_url": "...",
      "holding": "...",
      "affected_units": [{ "unit_ref": "UU-2009-4/pasal-149/ayat-2/huruf-b", "effect": "..." }]
    }]
  },
  "content": {
    "raw_text_path": "data/texts/UU_2_2024.txt",
    "sections": {
      "menimbang": "...",
      "mengingat": ["...", "..."],
      "memutuskan": "...",
      "penjelasan_umum": "...",
      "penjelasan_pasal_demi_pasal": [
        { "ref_unit_id": "UU-2024-2/.../pasal-5/ayat-1", "text": "Cukup jelas." }
      ],
      "lampiran": [{ "title": "...", "text": "..." }]
    },
    "tree": {  // hanya batang tubuh; tanpa amandemen
      "unit_type": "dokumen",
      "unit_id": "UU-2024-2",
      "label_display": "Dokumen",
      "children": [ "buku"/"bab"/"bagian"/"paragraf"/"pasal"/"ayat"/"huruf"/"angka" nodes
        // tiap node WAJIB punya local_content (unit only)
        // parent pointers: parent_pasal_id, parent_ayat_id, parent_huruf_id sesuai level
      ]
    },
    "amendments": [
      // daftar operasi perubahan yg dilakukan dokumen ini TERHADAP dokumen target lain
      // TIDAK menjadi child di 'tree'. Gunakan operasi:
      // 'hapus_pasal' | 'sisip_pasal' | 'ubah_pasal' | 'hapus_ayat' | 'sisip_ayat' | 'ubah_ayat' | 'hapus_huruf' | 'sisip_huruf' | 'ubah_huruf' | 'hapus_angka' | 'sisip_angka' | 'ubah_angka'
      { "seq": 17, "op": "hapus_pasal",
        "target_doc_id": "UU-XXXX-Y",
        "target_unit": { "unit_ref": "UU-XXXX-Y/pasal-27" },
        "note": "Pasal 27 dihapus" },
      { "seq": 18, "op": "sisip_pasal",
        "target_doc_id": "UU-XXXX-Y",
        "position": { "after": "UU-XXXX-Y/pasal-34" },
        "new_unit": { "unit_type": "pasal", "unit_id": "UU-XXXX-Y/pasal-34A", "number_label": "34A", "local_content": "..." },
        "note": "Disisipkan Pasal 34A" },
      { "seq": 2, "op": "ubah_pasal",
        "target_doc_id": "UU-XXXX-Y",
        "target_unit": { "unit_ref": "UU-XXXX-Y/pasal-87" },
        "new_text": "... (pasal setelah diubah)",
        "note": "Ketentuan Pasal 87 diubah ..." }
    ]
  }
}

VALIDATOR WAJIB:
- unit_id unik dan deterministik (berbasis path).
- parent_pasal_id untuk ayat; parent_ayat_id untuk huruf; parent_huruf_id untuk angka.
- number_label: pasal/ayat/angka = digit (allow A/B, bis/ter), huruf = a–z.
- citation_string ada di node pasal (dipakai luas).
- raw_text_path ada (file .txt).
- amendments: field wajib sesuai op; target_doc_id/target_unit valid.

============================================================
3) DATABASE & INDEKS (PAKAI MODEL YANG SUDAH ADA)
============================================================
- GIN index di legal_units.content_vector (TSVECTOR) (sudah).
- HNSW pgvector index di document_vectors.embedding (cosine) (sudah).
- Tambah (opsional) table ringan `legal_edges` untuk relasi semantik unit-level:
  (id, src_unit_id, dst_unit_id, rel_type ENUM['mendefinisikan','mengatur_detail','mengecualikan','sanksi','merujuk','mengubah','mencabut'], confidence, created_at).

Centralize SQL di src/db/queries.py:
- Explicit chain: doc → pasal → ayat → huruf → angka dengan parent_*.
- Sibling counters: count huruf pada ayat X; count ayat pada pasal Y; dll.
- FTS leaf-level: legal_units.bm25_body ts_rank + @@ tsquery('indonesian').
- Vector pasal-level: document_vectors (join ke legal_units untuk citation_string).
Semua query parameterized; tidak ada string format injection.

============================================================
4) EMBEDDING (Jina v4) & CACHING
============================================================
- Default: jina-embeddings-v4; TASK: indexing=retrieval.passage, query=retrieval.query.
- EMBEDDING_DIM default 1024 (izin 768 jika dikonfigurasi).
- LRU cache untuk query embeddings (in-memory).
- Semua HTTP via src/utils/http.py (timeout, retries, jitter, circuit breaker).
- Validasi dimensi kolom pgvector pada startup; error jelas jika mismatch.

============================================================
5) RETRIEVAL ROUTER + PARALEL (FTS + VECTOR) + RERANK
============================================================
Router (src/services/retriever/hybrid_retriever.py):
- Explicit patterns (dok lengkap):
  1) Dokumen + pasal/ayat/huruf/angka:
     (?:(UU|Undang-Undang|PP|Perpres|Perpu|Permen|POJK|SEOJK|SEMA|SEMK|Perda))\s*(?:No|Nomor)?\s*(\d+)[/ ](\d{4})(?:.*?pasal\s+(\d+[A-Z]?)(?:\s+ayat\s+\(?(\d+)\)?)?(?:\s+huruf\s+([a-z]))?(?:\s+angka\s+(\d+))?)?
  2) Pasal first:
     pasal\s+(\d+[A-Z]?)(?:\s+ayat\s+\(?(\d+)\)?)?(?:\s+huruf\s+([a-z]))?(?:\s+angka\s+(\d+))?
- Jika cocok → explicit pipeline (SQL only).
- Jika tidak → contextual pipeline.

Contextual Pipeline:
- Jalankan FTS (leaf) **dan** Vector (pasal) **secara paralel** (thread/async) dengan timeout (RETRIEVAL_TIMEOUT_S).
- Ambil masing-masing K (RETRIEVAL_FTS_K, RETRIEVAL_VECTOR_K).
- Merge unique by unit_id; normalisasi skor sumber (min–max atau z-score).
- Rerank via Jina reranker (RERANK_TIMEOUT_S). Jika timeout/error → fallback blended sort: 0.6*vector + 0.4*fts.
- Ambil FINAL_TOP_K.

Explicit Pipeline:
- Resolve doc (form/number/year) → pasal → ayat → huruf → angka via parent_* deterministic joins.
- Menjawab "ada berapa ayat/huruf/angka" dengan sibling counters.
- Boleh rerank jika ada multi-kandidat (versi/edisi), selain itu preserve order.

============================================================
6) FRAMEWORK ADAPTERS (ORCHESTRATION ONLY)
============================================================
Buat src/services/retriever/framework_adapters.py:
- SqlFTSRetriever: panggil SQL FTS internal; return candidates {unit_id,text,score,meta}.
- PgVectorRetriever: panggil SQL vector internal; return idem.
- ParallelRetriever: jalankan kedua retriever; merge hasil.
- JinaRerankNode: wrapper Jina rerank API (timeout, retries, circuit fallback).
Framework didukung: Haystack / LangChain / LlamaIndex.
CATATAN: Framework TIDAK menyimpan dokumen (storage tetap Postgres + pgvector + FTS).

============================================================
7) RERANKER HARDENING
============================================================
src/services/search/reranker.py:
- Jina v2 multilingual client (sudah ada). Tambahkan:
  - Circuit breaker: jika 2x timeout beruntun dlm 60s → open 30s (skip external).
  - Structured logs: {input_count, output_count, duration_ms, attempts}.
  - Fallback: blended score.

============================================================
8) LLM ROUTER & PROMPTS (INTENT-AWARE)
============================================================
Router (src/services/llm/router.py):
- Provider switch: gemini (default) | gpt | claude.
- Method:
  generate(system_prompt, user_prompt, tools=None, max_tokens=settings.LLM_MAX_OUTPUT_TOKENS, temperature=0.1, timeout=settings.LLM_TIMEOUT_S) -> {text, tokens_in, tokens_out, model, latency_ms}
- Wajib: 429 backoff + jitter; hard timeout; optional streaming (off default).

Prompts (src/services/llm/prompts.py):
- Semua template WAJIB: 
  (a) larangan halusinasi; 
  (b) kutip literal dari 'quotes'; 
  (c) jika konteks kurang, katakan demikian; 
  (d) OUTPUT JSON DULU, lalu human.

INTENTS (template khusus):
- T-ExplicitCite (kutip pasal/ayat/huruf/angka)
- T-Definition (definisi & dasar hukum)
- T-Procedure (langkah, pelaksana, batas waktu, dokumen)
- T-Sanction (jenis sanksi, ancaman, subjek)
- T-Exception (pengecualian, klausul “sebagaimana dimaksud”)
- T-CrossRef (pendefinisi/penerapan/sanksi terkait)
- T-AmendmentTrace (riwayat perubahan, status berlaku)
- T-Count (jumlah ayat/huruf/angka)
- T-MetaCharacter (tanda baca & implikasi)
- T-Authority (kewenangan siapa melakukan apa)
- T-Obligation/Right (kewajiban/hak)
- T-Transitional (ketentuan peralihan/penutup)

Tambahkan 6 TEMPLATE “HURUF/KARAKTER” (gunakan verbatim, siap pakai):
1) Template Ringkas: [huruf/karakter literal] + [kalimat konteks] + [makna hukum] + [sumber lengkap].
2) Ekspansi Konseptual: huruf sebagai butir ayat; makna & relasi dgn huruf (a)–(d).
3) Perbandingan Redaksi: versi lama vs baru, signifikansi.
4) Penomoran Hierarkis: Pasal → Ayat → Huruf → Angka; asas lex specialis.
5) Penjelasan Umum/Pasal: tautkan ke Penjelasan.
6) Meta-karakter: koma/titik-koma/dash & fungsinya.

OUTPUT JSON (STRICT):
{
  "answer_text": "...",
  "intent": "explicit_cite|definition|procedure|sanction|exception|count|crossref|amendment_trace|meta_character|authority|obligation|right|transitional",
  "strategy": "explicit|hybrid",
  "quotes": [{ "unit_id":"...", "unit_type":"pasal|ayat|huruf|angka", "text":"..." }],
  "citations": [{ "doc_form":"UU","doc_number":"2","doc_year":2024,"unit_type":"ayat","unit_id":"...","label_display":"Pasal 5 ayat (1)"}],
  "reasoning_trace": "...",
  "confidence": 0.0
}
Setelah JSON, tampilkan versi human: "Jawaban — … | Dasar Hukum — … | Catatan — … | Keterbatasan — …"

============================================================
9) ANSWER BUILDER (JSON-FIRST)
============================================================
src/services/answers/answer_builder.py:
- Input: query, results[], intent, strategy, llm_router.
- Build grounding: kutipan literal + 1–2 baris konteks di sekitar.
- Pilih template berdasar intent; panggil LLM; validasi JSON (sekali retry jika invalid).
- Pastikan setiap citation berisi unit_id + doc_form/number/year.
- Heuristik confidence:
  0.5 * normalized_rerank + 0.3 * (overlap FTS∩Vector) + 0.2 * explicit_match_flag.

============================================================
10) AMANDEMEN & CROSS-REF (LIGHT, BERMANFAAT)
============================================================
- Simpan daftar operasi di content.amendments (JSONB).
- (Opsional) materialize "edisi terkonsolidasi" untuk doc target (apply patch) agar query eksplisit/ count mengikuti keadaan terkini.
- legal_edges (opsional) untuk relasi semantik, dibangun heuristik dari frasa:
  "sebagaimana dimaksud", "menurut", "berdasarkan", "diatur dalam", "sesuai dengan".
- AmendmentTrace intent: tampilkan lineage: X → mengubah Y → Y dicabut Z, dengan tanggal/efektivitas.

============================================================
11) INTENT CLASSIFIER RINGAN
============================================================
Hybrid retriever: regex + keyword routing:
- explicit_cite: pola dokumen + pasal/ayat/huruf/angka (lihat §5).
- count: "ada berapa (ayat|huruf|angka)".
- sanction: "dipidana", "denda", "sanksi administratif".
- procedure: "tahapan", "prosedur", "syarat", "batas waktu".
- definition: "yang dimaksud", "definisi", "adalah".
- exception: "dikecualikan", "kecuali", "pengecualian".
- authority/obligation/right: "berwenang", "wajib", "hak".
- transitional: "ketentuan peralihan/penutup".
Jika explicit → SQL path; selain itu → hybrid parallel.

============================================================
12) OBSERVABILITY & SAFETY
============================================================
- Structured logs setiap tahap: {query, strategy, fts_hits, vec_hits, merged, reranked, duration_ms}.
- HTTP logs: {url, method, status, duration_ms, attempts}; LLM: {provider, latency_ms, tokens_in/out}.
- Circuit breaker di reranker & Jina embeddings; fallback jalan tanpa crash.
- Token budgeting LLM: max output tokens via settings.

============================================================
13) KINERJA (BUDGET)
============================================================
- Explicit median (warm) < 800 ms.
- Contextual median (warm) < 2.5 s (sudah termasuk rerank).
- Rerank timeout ≤ 1.2 s; LLM timeout ≤ 15 s.
- Bila API eksternal timeout → tetap beri jawaban (atau "Tidak cukup konteks..."), jangan error.

============================================================
14) PENGGUNAAN FRAMEWORK (WAJIB MODULAR)
============================================================
- Haystack/LangChain/LlamaIndex hanya sebagai ORCHESTRATOR untuk contextual retrieval & rerank.
- Simpan adapter tipis (≤200 LOC per retriever/node).
- Storage TETAP Postgres + pgvector + TSVECTOR (bukan milik framework).

============================================================
15) FILES TO TOUCH / ADD (≤300 LOC/file)
============================================================
- src/config/settings.py                 (extend keys: retrieval K, timeouts, provider)
- src/validators/json_v2_validator.py    (NEW)
- src/schemas/json_v2.md                 (NEW spec)
- src/db/queries.py                      (NEW explicit/FTS/vector/counters SQL)
- src/services/retriever/hybrid_retriever.py (router + parallel + fallbacks)
- src/services/retriever/framework_adapters.py (NEW; Haystack/LangChain/LlamaIndex)
- src/services/search/hybrid_search.py   (tipis; orkestra saja)
- src/services/search/reranker.py        (circuit + fallback)
- src/services/embedding/embedder.py     (Jina v4 stable + LRU)
- src/services/llm/router.py             (provider switch + backoff)
- src/services/llm/prompts.py            (intent templates + 6 huruf patterns)
- src/services/answers/answer_builder.py (JSON-first + validator)
- src/utils/http.py                      (retries/jitter/circuit)
- tests/unit/*, tests/e2e/*, tests/run_tests.py (per acceptance)

============================================================
16) SETTINGS (EXTEND)
============================================================
# Retrieval
RETRIEVAL_FTS_K=40
RETRIEVAL_VECTOR_K=40
MERGE_TOP_K=60
FINAL_TOP_K=10

# Timeouts/Concurrency
QUERY_TIMEOUT_S=12
RETRIEVAL_TIMEOUT_S=8
RERANK_TIMEOUT_S=10
HTTP_TIMEOUT_S=20
PARALLEL_MAX_WORKERS=4

# Embeddings
EMBEDDING_MODEL="jina-embeddings-v4"
EMBEDDING_DIM=1024
TASK_QUERY="retrieval.query"
TASK_PASSAGE="retrieval.passage"

# Reranker
ENABLE_RERANKER=True
RERANKER_MODEL="jina-reranker-v2-base-multilingual"
RERANK_TOP_N=10

# Framework
FRAMEWORK="haystack"  # or "langchain" | "llamaindex"

# LLM
LLM_PROVIDER="gemini"  # or "gpt" | "claude"
LLM_TIMEOUT_S=15
LLM_MAX_OUTPUT_TOKENS=1024

============================================================
17) TEST PLAN (WAJIB HIJAU)
============================================================
Unit:
- test_query_router.py: ekstraksi (UU 4/2009 Pasal 149 ayat (2) huruf b), variasi spasi/case, semua jenis dokumen.
- test_explicit_sql.py: resolve huruf via parent_ayat_id, angka via parent_huruf_id; counter "ada berapa ...".
- test_parallel_retrieval.py: FTS+Vector paralel; merge unik; hormati timeout.
- test_reranker.py: happy path; timeout → fallback; circuit open/close.
- test_answer_builder.py: semua template valid JSON; citations mencakup unit_id.
- test_llm_router.py: backoff/timeout.

E2E (smoke):
- "apa isi pasal 149 ayat (2) huruf b uu 4/2009" → kutipan huruf b literal + breadcrumb + citation unit_id.
- "ada berapa huruf dalam pasal 5 ayat (1) uu 2/2024" → count + daftar (a, b, c, …).
- "uu 2/2024 mengatur tentang apa?" → ringkasan ruang lingkup + top citations; tanpa 429 pecah ke user.
- "apa itu kewenangan khusus?" → hybrid parallel, reranked, definisi dengan sitasi multi-dokumen jika relevan.

Perf:
- tests/run_tests.py --quick → 100% pass dalam waktu wajar.

============================================================
18) ACCEPTANCE
============================================================
- Tidak ada 400/429/timeout eksternal yang bocor ke user; fallback selalu aktif.
- Explicit query selalu resolve node tepat (huruf/angka) dan mengutip literal.
- Contextual query SELALU menjalankan FTS+Vector paralel dan (jika aktif) rerank.
- Semua jawaban menyertakan citations ber-anchored unit_id dan reasoning_trace singkat.
- Unit + E2E hijau. Log menunjukkan {strategy, timings, counts}.

============================================================
19) LANGKAH IMPLEMENTASI (URUT WAJIB)
============================================================
1. Tambah JSON V2 writer di extractor + validator; jangan hapus V1 dulu.
2. Buat src/db/queries.py (explicit/FTS/vector/sibling counters).
3. Refactor hybrid_retriever → router + parallel + fallbacks.
4. Tambah framework_adapters (pilih Haystack default).
5. Kuatkan reranker (circuit + fallback).
6. Tambah llm/router + llm/prompts + answers/answer_builder.
7. Tambah/benahi tests (unit & e2e).
8. Aktifkan structured logs; verifikasi SLA kinerja.

============================================================
20) EDGE CASES WAJIB DITANGANI
============================================================
- Bab roman (I–XX) & Buku opsional (KUHP/KUHAP).
- Suffix pasal: 14A, 27B; juga "bis", "ter".
- Penomoran leaf: huruf a–z; angka 1).
- Penjelasan Umum & Pasal demi Pasal (map ke ref_unit_id).
- “sebagaimana dimaksud pada Pasal/ayat/huruf …” → cross-ref detection.
- Status berlaku berdasarkan tanggal efektif; jika amandemen/mk mempengaruhi, tampilkan status.
- Konsolidasi (edisi terkini) saat explicit diminta versi terbaru (opsional via materialization).

DO IT NOW. Keep each diff small and focused. Document every public method with concise docstrings. 
If ambiguous, prefer: more deterministic, simpler, less redundancy.