Struktur Database (DDL ringkas & jelas)

1) Tabel dokumen (metadata induk)
CREATE TABLE legal_documents (
  id                  UUID PRIMARY KEY,                 -- UUID deterministik (saran: UUIDv5 dari form-number-year)
  source              TEXT NOT NULL,                    -- BPK, DJPP, dll
  type                TEXT NOT NULL,                    -- "Peraturan Perundang-undangan"
  form                TEXT NOT NULL,                    -- "UU", "PP", "Perpres", dst.
  form_short          TEXT NOT NULL,                    -- biasanya sama dgn form
  number              TEXT NOT NULL,                    -- nomor dokumen (string)
  year                INT  NOT NULL,                    -- tahun
  title               TEXT NOT NULL,                    -- judul lengkap
  teu                 TEXT,                             -- "Indonesia, Pemerintah Pusat"
  place_enacted       TEXT,
  date_enacted        DATE,
  date_promulgated    DATE,
  date_effective      DATE,
  status              TEXT NOT NULL DEFAULT 'Berlaku',
  language            TEXT DEFAULT 'Bahasa Indonesia',
  location            TEXT,
  field               TEXT,
  subject             JSONB DEFAULT '[]'::jsonb,        -- ["HUKUM ...", ...]
  relationships       JSONB DEFAULT '{}'::jsonb,        -- boleh disimpan mentah juga
  detail_url          TEXT,
  source_url          TEXT,
  pdf_url             TEXT,
  uji_materi_pdf_url  TEXT,
  pdf_path            TEXT,
  content             TEXT,                             -- opsional: full text dokumen (mentah/bersih)
  content_length      INT,
  processing_status   TEXT DEFAULT 'pending',
  error_message       TEXT,
  source_sha256       TEXT,
  pdf_sha256          TEXT,
  content_sha256      TEXT,
  meta                JSONB DEFAULT '{}'::jsonb,
  created_at          TIMESTAMPTZ DEFAULT now(),
  updated_at          TIMESTAMPTZ DEFAULT now(),
  processed_at        TIMESTAMPTZ
);

CREATE UNIQUE INDEX uq_doc_form_no_year ON legal_documents(form, number, year);
CREATE INDEX idx_doc_subject  ON legal_documents USING GIN (subject);

2) Tabel unit isi (hirarki + FTS)
CREATE TABLE units (
  unit_id         TEXT PRIMARY KEY,                        -- ex: "UU-2015-1/pasal-10/ayat-2/huruf-b"
  doc_id          UUID NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
  unit_type       TEXT NOT NULL,                           -- 'bab','bagian','paragraf','pasal','ayat','huruf','angka'
  number_label    TEXT,                                    -- label tampil: "1A","IA","1","a","(2)"
  ordinal_int     INT,                                     -- numerik: 1 pada 1A / IA
  ordinal_suffix  TEXT,                                    -- "A" pada 1A / IA
  label_display   TEXT NOT NULL,                           -- "Pasal 1A","(2)","a."
  seq_sort_key    TEXT,                                    -- ex: "0001|A" (ORDER BY stabil)
  -- Konten:
  content         TEXT,                                    -- KHUSUS PASAL: isi lengkap pasal (termasuk ayat/huruf/angka)
  local_content   TEXT,                                    -- KHUSUS LEAF: isi ayat/huruf/angka
  display_text    TEXT,                                    -- KHUSUS LEAF: label + isi (siap tampil)
  bm25_body       TEXT,                                    -- KHUSUS LEAF: bahan FTS (biasanya = local_content)
  span            JSONB,                                   -- posisi leaf di 'content' pasal: {"start":int,"end":int}
  parent_pasal_id TEXT,                                    -- anchor balik ke pasal induk
  path            JSONB,                                   -- breadcrumb: [{type,label,unit_id}, ...]
  citation_string TEXT,                                    -- "UU X/XXXX, Pasal 10 ayat (2) huruf b"
  tags_semantik   TEXT[],                                  -- ["definisi","sanksi","kewajiban",...]
  entities        TEXT[],                                  -- hasil NER (opsional)
  created_at      TIMESTAMPTZ DEFAULT now(),
  -- FTS generated:
  bm25_tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(bm25_body,''))) STORED
);

-- Indexing
CREATE INDEX idx_units_bm25_tsv   ON units USING GIN (bm25_tsv);
CREATE INDEX idx_units_bm25_trgm  ON units USING GIN (bm25_body gin_trgm_ops);
CREATE INDEX idx_units_doc_seq    ON units(doc_id, unit_type, ordinal_int, ordinal_suffix);
CREATE INDEX idx_units_parent     ON units(parent_pasal_id);
CREATE INDEX idx_units_path       ON units USING GIN (path);

3) Tabel vektor (pasal-only, embedding dari isi lengkap pasal)
CREATE TABLE document_vectors (
  id                 TEXT PRIMARY KEY,                         -- gunakan unit_id pasal (ex: "UU-2015-1/pasal-10")
  document_id        UUID NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
  content_type       TEXT NOT NULL DEFAULT 'pasal',            -- selalu 'pasal'
  content_text       TEXT NOT NULL,                            -- isi lengkap pasal (backup/debug)
  embedding          VECTOR(768) NOT NULL,                     -- hasil embedding dari content_text
  -- Filter & navigasi:
  doc_type           TEXT NOT NULL,                            -- "UU","PP",...
  doc_year           INT  NOT NULL,
  doc_number         TEXT NOT NULL,
  doc_status         TEXT NOT NULL,
  bab_ordinal_int    INT,
  bab_suffix         TEXT,
  pasal_ordinal_int  INT NOT NULL,
  pasal_suffix       TEXT,
  subjects           JSONB DEFAULT '[]'::jsonb,
  tags_semantik      JSONB DEFAULT '[]'::jsonb,
  hierarchy_path     TEXT,
  token_count        INT DEFAULT 0,
  char_count         INT DEFAULT 0,
  embedding_model    TEXT DEFAULT 'text-embedding-004',
  embedding_version  TEXT DEFAULT 'v1',
  created_at         TIMESTAMPTZ DEFAULT now(),
  updated_at         TIMESTAMPTZ DEFAULT now()
);

-- ANN index (ivfflat); adjust lists/probes di session
CREATE INDEX idx_docvec_ann ON document_vectors
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);

CREATE INDEX idx_docvec_filters ON document_vectors (doc_type, doc_year, doc_status);
CREATE INDEX idx_docvec_order   ON document_vectors (pasal_ordinal_int, pasal_suffix);


	Parser JSON “master” (ekstraksi → strukturisasi) menghasilkan struktur pasal + anak sampai huruf/angka sesuai skema di bawah.
	2.	Migrasi DB (SQLAlchemy + migration) ke skema MVP:
	•	legal_documents (metadata)
	•	units (struktur isi; FTS di leaf)
	•	document_vectors (pasal-only; embedding dari content pasal)
	3.	Indexer:
	•	Embed content dari semua pasal → simpan ke document_vectors.
	•	FTS index untuk units leaf: bm25_tsv (tsvector) + trigram.
	4.	Query router:
	•	Pola eksplisit (“Pasal 10 ayat (2) huruf b”) → direct lookup units.unit_id.
	•	Pencarian semantik → ANN top-k pasal → FTS lokal leaf pada pasal kandidat → hasil final + sitasi.
	5.	Tes end-to-end:
	•	10 contoh query eksplisit & semantik lulus (lihat bagian Acceptance Tests).
	Skema DB (MVP, anti-overengineering)

1) legal_documents
		•	id (UUID, deterministik: v5 dari form-number-year)
		•	form, number, year, title, status, subject (JSONB), urls (opsional)
		•	created_at, updated_at

2) units
		•	unit_id (PK, TEXT) — ex: UU-2015-1/pasal-10/ayat-2/huruf-b
		•	doc_id (FK → legal_documents.id)
		•	unit_type ('pasal'|'ayat'|'huruf'|'angka'|'bab'|'bagian'|'paragraf')
		•	number_label, ordinal_int, ordinal_suffix, label_display
		•	pasal: content (TEXT)
		•	leaf: local_content, display_text, bm25_body, parent_pasal_id (TEXT)
		•	path (JSONB), citation_string (TEXT)
		•	generated column: bm25_tsv = to_tsvector('simple', coalesce(bm25_body,''))
		•	index:
		•	GIN(bm25_tsv), GIN(bm25_body gin_trgm_ops)
		•	(doc_id, unit_type, ordinal_int, ordinal_suffix)
		•	parent_pasal_id

3) document_vectors
		•	id (TEXT, = unit_id pasal)
		•	document_id (FK → legal_documents.id)
		•	content_type = ‘pasal’
		•	content_text (TEXT) — salinan content pasal (untuk debug/backup)
		•	embedding VECTOR(768)
		•	doc_type, doc_year, doc_number, doc_status
		•	pasal_ordinal_int, pasal_suffix
		•	(opsional) subjects JSONB
		•	index:
		•	ivfflat(embedding vector_cosine_ops) WITH (lists=200)
		•	(doc_type, doc_year, doc_status)

Embedding hanya dibuat untuk PASAL dari content penuh.
FTS hanya di leaf (ayat/huruf/angka).

⸻

🧠 Perubahan/Penyesuaian ke Kode yang Ada (Refactor Plan)
		1.	Parser
		•	Outputkan JSON master sesuai kontrak di atas.
		•	Pastikan:
		•	unit_id pathful dan stabil.
		•	Pasal punya content lengkap.
		•	Leaf punya local_content/display_text/bm25_body.
		•	Sisipan (1A/IA) di-handle via number_label + ordinal_int + ordinal_suffix.
		2.	Loader JSON → DB
		•	Insert/Upsert legal_documents (gunakan UUID v5 dari form-number-year).
		•	Traverse document_tree:
		•	Insert pasal ke units (content diisi).
		•	Insert leaf ke units (bm25_body, display_text, parent_pasal_id).
		•	Build bm25_tsv via generated column (lihat skema).
		3.	Embedding Indexer
		•	Untuk setiap pasal: ambil content → bikin embedding → simpan ke document_vectors
		•	id = unit_id pasal
		•	filter fields (doc_type/year/number/status)
		•	ANN index: ivfflat (lists=200). Query pakai SET ivfflat.probes=10.
		4.	Query Engine
		•	Router eksplisit:
		•	Regex pattern: Pasal (?P<pasal>\w+)( ayat \((?P<ayat>\d+)\))?( huruf (?P<huruf>[a-z]))?( angka (?P<angka>\d+))?
		•	Compose unit_id → SELECT * FROM units WHERE unit_id=:id (O(1)).
		•	Semantik:
		1.	ANN ke document_vectors (filter metadata jika ada di query; mis. form/year/number/status).
		2.	Untuk tiap pasal kandidat → FTS lokal:
		SELECT unit_id, display_text, citation_string
FROM units
WHERE parent_pasal_id = :pasal_id
ORDER BY bm25_tsv @@ plainto_tsquery('simple', :q) DESC,
         similarity(bm25_body, :q) DESC
LIMIT 3;



🧪 Acceptance Tests (WAJIB LULUS)
	1.	Eksplisit
	•	Input: “Apa isi Pasal 10 ayat (2) huruf b UU X Tahun Y?”
	•	Output: teks leaf yang tepat + citation_string + dokumen benar.
	2.	Definisi
	•	Input: “Apa definisi Penyidikan dalam KUHAP?”
	•	Flow: ANN top-k pasal definisi → leaf pick angka definisi yang tepat → tampilkan kutipan.
	3.	Semantik umum → presisi
	•	Input: “apakah pendaftaran cagar budaya dipungut biaya?”
	•	Flow: ANN → Pasal 10 → leaf pick ayat (2) → “tidak dipungut biaya”.
	4.	Sisipan
	•	Input: “Apa isi Pasal 11A?”
	•	Pastikan sort & lookup 11A benar (bukan 11 atau 12).
	5.	Dokumen filter
	•	Input: “apa isi pasal 23 UU No 20 Tahun 2023?”
	•	ANN + filter doc_number/doc_year → hasil dari dokumen yang benar.
	6.	Fallback panjang
	•	Jika satu pasal terlalu panjang dan di-split per ayat untuk embedding (edge case), hasil tetap benar & mengembalikan satu pasal sebagai konteks (group by pasal).
