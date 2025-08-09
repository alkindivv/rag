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

# USER→ASSISTANT: Jawaban dengan kutipan & ringkasan
ANSWER_WITH_CITATIONS = (
    "Pertanyaan pengguna:\n"
    "{question}\n\n"
    "Konteks (potongan hasil pencarian):\n"
    "{contexts}\n\n"
    "Instruksi:\n"
    "- Gunakan hanya konteks di atas.\n"
    "- Kutip unit paling spesifik (mis. Pasal X ayat (Y) huruf Z) menggunakan `citation_string` yang diberikan.\n"
    "- Tulis jawaban dalam bahasa Indonesia yang jelas. Jika aturan berubah karena ‘diubah/dicabut’, jelaskan singkat.\n"
    "- Jika tidak cukup data, jawab: 'Tidak ditemukan dalam konteks yang diberikan.'\n"
)

# Reformulasi query untuk retrieval
QUERY_REWRITE = (
    "Ubah pertanyaan berikut menjadi query pencarian yang lebih tepat untuk hukum Indonesia.\n"
    "Fokus pada kata kunci normatif (mis. 'izin', 'sanksi', 'kewenangan', 'kewajiban'), entitas (UU/PP/Perpres),\n"
    "dan unit (Pasal/Ayat/Huruf/Angka). Jangan menambahkan opini.\n\n"
    "Pertanyaan: {question}\n"
    "Hasil: "
)

# Ringkasan pasal/ayat
SUMMARIZE_UNITS = (
    "Ringkas poin kunci dari unit berikut menjadi bullet list.\n"
    "Pertahankan istilah hukum. Jangan mengubah makna.\n\n"
    "{contexts}\n\n"
    "Hasil ringkas:"
)

# Verifikasi / kritik jawaban (self‑check)
CRITIQUE = (
    "Periksa jawaban berikut terhadap konteks hukum yang disediakan.\n"
    "- Tandai klaim tanpa dasar.\n"
    "- Tunjukkan jika kutipan tidak sesuai unit.\n"
    "- Beri saran perbaikan singkat.\n\n"
    "Jawaban:\n{answer}\n\n"
    "Konteks:\n{contexts}\n\n"
    "Laporan:"
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

# Helper kecil untuk mencetak contexts ke template
def join_contexts(results, max_len: int = 6) -> str:
    """
    results: List[dict] dengan field minimal:
      - 'content' (teks potongan)
      - 'citation' atau 'citation_string'
      - 'unit_id' (opsional)
    """
    rows = []
    for i, r in enumerate(results[:max_len], 1):
        cit = r.get("citation") or r.get("citation_string") or "-"
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > 800:
            content = content[:800] + "…"
        rows.append(f"[{i}] {cit}\n{content}\n")
    return "\n".join(rows)