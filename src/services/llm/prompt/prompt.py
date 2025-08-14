#!/usr/bin/env python3
"""
llm/prompts.py
Kumpulan prompt template (singkat, tajam) untuk RAG hukum Indonesia.
Semua template memakai format .format() / f-string style (tanpa dependency tambahan).
"""

# SYSTEM: Jalankan guardrails umum
SYSTEM_RAG = (
    "Anda adalah asisten hukum Indonesia yang teliti.\n"
    "- Jawab ringkas, tepat, dan mengutip pasal/ayat/huruf secara eksplisit.\n"
    "- Jangan mengarang. Jika ragu atau tidak ada di dokumen, katakan tidak ditemukan.\n"
    "- Format kutipan: “{citation_string}”.\n"
    "- Jika ada beberapa pasal relevan, urutkan dari yang paling spesifik ke umum.\n"
)

# USER→ASSISTANT: Jawaban dengan kutipan & ringkasan (Enhanced for better focus)
ANSWER_WITH_CITATIONS = (
    "Pertanyaan pengguna:\n"
    "{question}\n\n"
    "Konteks hukum yang tersedia:\n"
    "{contexts}\n\n"
    "INSTRUKSI PRIORITAS:\n"
    "1. FOKUS TOTAL pada pertanyaan spesifik yang diajukan\n"
    "2. ANALISIS jenis pertanyaan:\n"
    "   - Jika tanya 'pasal apa/UU apa' → sebutkan EKSPLISIT pasal dan nomor UU\n"
    "   - Jika tanya 'definisi/pengertian' → berikan definisi TEPAT dari konteks\n"
    "   - Jika tanya 'sanksi/pidana' → sebutkan JENIS sanksi dan besarannya\n"
    "   - Jika tanya 'prosedur/cara' → jelaskan LANGKAH-LANGKAH yang tepat\n"
    "   - Jika tanya 'siapa/kewenangan' → identifikasi PIHAK yang berwenang\n"
    "3. GUNAKAN HANYA informasi dari konteks di atas - DILARANG menambah dari pengetahuan umum\n"
    "4. FORMAT JAWABAN:\n"
    "   a) Jawaban LANGSUNG dalam 1-2 kalimat pertama\n"
    "   b) Kutipan pasal lengkap: \"Berdasarkan [citation_string yang diberikan]\"\n"
    "   c) Penjelasan tambahan HANYA jika diminta\n"
    "5. KRITERIA KUALITAS:\n"
    "   - Jawaban HARUS menjawab pertanyaan eksplisit\n"
    "   - HARUS ada kutipan citation_string yang tepat\n"
    "   - HINDARI informasi umum yang tidak diminta\n"
    "6. Jika konteks TIDAK cukup untuk menjawab pertanyaan spesifik:\n"
    "   'Berdasarkan konteks yang tersedia, informasi spesifik mengenai [topik yang ditanyakan] tidak dapat dijawab secara lengkap.'\n\n"
    "MULAI JAWABAN:"
)

# Reformulasi query untuk retrieval yang lebih efektif
QUERY_REWRITE = (
    "Analisis pertanyaan hukum berikut dan ubah menjadi query pencarian yang optimal:\n\n"
    "Pertanyaan asli: {question}\n\n"
    "PANDUAN REFORMULASI:\n"
    "1. Identifikasi INTENT utama (definisi/sanksi/prosedur/kewenangan/dll)\n"
    "2. Ekstrak ENTITAS hukum (UU/PP/Perpres, nomor, tahun)\n"
    "3. Fokus pada KATA KUNCI normatif (izin, sanksi, kewajiban, hak, prosedur)\n"
    "4. Pertahankan UNIT spesifik (Pasal, Ayat, Huruf, Angka)\n"
    "5. Hilangkan kata tanya umum, fokus pada substansi\n\n"
    "Query hasil reformulasi:"
)

# Ringkasan pasal/ayat yang efektif
SUMMARIZE_UNITS = (
    "TUGAS: Buat ringkasan fokus untuk menjawab pertanyaan pengguna\n\n"
    "Konteks hukum:\n"
    "{contexts}\n\n"
    "ATURAN RINGKASAN:\n"
    "1. PERTAHANKAN istilah hukum original - jangan parafrase\n"
    "2. PRIORITASKAN informasi yang langsung menjawab pertanyaan\n"
    "3. FORMAT: Bullet point dengan struktur:\n"
    "   • [Citation lengkap]: [Inti substansi]\n"
    "4. URUTAN: Dari yang paling spesifik ke umum\n"
    "5. BATASI: Maksimal 5 poin utama\n\n"
    "Ringkasan terstruktur:"
)

# Verifikasi / kritik jawaban (self‑check) yang ketat
CRITIQUE = (
    "AUDIT KUALITAS JAWABAN HUKUM\n\n"
    "Jawaban yang akan diaudit:\n{answer}\n\n"
    "Konteks referensi:\n{contexts}\n\n"
    "CHECKLIST VERIFIKASI:\n"
    "1. AKURASI KUTIPAN:\n"
    "   ☐ Apakah semua citation_string akurat?\n"
    "   ☐ Apakah pasal/ayat yang dikutip benar?\n"
    "2. RELEVANSI JAWABAN:\n"
    "   ☐ Apakah jawaban langsung menjawab pertanyaan?\n"
    "   ☐ Apakah ada informasi tidak relevan?\n"
    "3. KELENGKAPAN:\n"
    "   ☐ Apakah ada informasi penting yang terlewat?\n"
    "   ☐ Apakah semua klaim memiliki dasar hukum?\n"
    "4. KONSISTENSI:\n"
    "   ☐ Apakah tidak ada kontradiksi internal?\n\n"
    "HASIL AUDIT:"
)

# Generator format kutipan (opsional jika ingin diseragamkan di LLM)
CITATION_FORMATTER = (
    "Berdasarkan path: {path}, bentuklah citation_string ringkas seperti:\n"
    "\"{doc_title}, Pasal X ayat (Y) huruf Z\" (opsional angka),\n"
    "atau hanya \"{doc_title}, Pasal X\" jika tanpa ayat/huruf/angka.\n"
    "Hasil hanya string final tanpa penjelasan."
)

# Jawaban JSON terstruktur (untuk UI)
STRUCTURED_JSON_ANSWER = (
    "Kembalikan jawaban sebagai JSON dengan schema:\n"
    "{{\n"
    '  "answer": "<teks jawaban singkat>",\n'
    '  "citations": [\n'
    '     {{"unit_id":"...", "citation_string":"...", "doc_identifier":"..."}},\n'
    "  ],\n"
    '  "notes": "<opsional, klarifikasi singkat>"\n'
    "}}\n\n"
    "Pertanyaan: {question}\n"
    "Konteks:\n"
    "{contexts}\n"
)

# Prompt untuk menjahit beberapa potongan jadi narasi koheren
COMPOSE_FINAL = (
    "Susun jawaban final yang koheren dari potongan berikut tanpa menambah informasi baru.\n"
    "Setiap klaim harus ditopang oleh kutipan yang tersedia.\n\n"
    "{bullets}\n\n"
    "Jawaban final singkat:"
)

# Prompt untuk debug (jejak keputusan singkat)
EXPLAIN_RETRIEVAL_DECISIONS = (
    "Berikut log retrieval (FTS/dense/reranker). Jelaskan singkat kenapa hasil teratas relevan.\n\n"
    "{logs}\n"
    "Penjelasan (maks 5 kalimat):"
)

# Helper untuk format konteks yang optimal
def join_contexts(results, max_len: int = 6) -> str:
    """
    Enhanced context formatting for better LLM processing.

    Args:
        results: List[dict] dengan field minimal:
          - 'content' (teks potongan)
          - 'citation' atau 'citation_string'
          - 'unit_id' (opsional)
          - 'score' (opsional, untuk prioritas)
        max_len: Maksimal jumlah konteks

    Returns:
        Formatted context string untuk prompt
    """
    if not results:
        return "Tidak ada konteks yang tersedia."

    rows = []
    for i, r in enumerate(results[:max_len], 1):
        # Get citation with fallback
        cit = r.get("citation") or r.get("citation_string") or f"[Referensi {i}]"

        # Clean and truncate content
        content = (r.get("content") or "").strip()
        content = content.replace("\n", " ").replace("\t", " ")

        # More aggressive truncation for focus
        if len(content) > 600:
            content = content[:600] + "..."

        # Add relevance indicator if available
        score_info = ""
        if r.get("score") and isinstance(r["score"], (int, float)):
            score_info = f" (relevansi: {r['score']:.2f})"

        # Format with clear structure
        rows.append(f"[{i}] SUMBER: {cit}{score_info}\nISI: {content}\n")

    return "\n".join(rows)
