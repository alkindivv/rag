SYSTEM_PROMPT = """Anda adalah asisten hukum Indonesia.
- Jawab ringkas, akurat, dan sertakan kutipan sumber pada akhir kalimat berupa [citation].
- Jangan mengarang pasal. Jika tidak yakin, bilang tidak yakin dan sarankan pasal terkait."""

def format_user_prompt(query: str, context: str) -> str:
    return f"""Pertanyaan: {query}

Berikut adalah potongan relevan dari peraturan perundang-undangan:
{context}

Instruksi:
- Gunakan potongan di atas untuk menjawab.
- Setiap klaim normatif cantumkan [citation] dari potongan yang digunakan.
- Jika pertanyaan meminta isi lengkap pasal/ayat/huruf/angka, kutip persis bagian tersebut lalu beri [citation].
Jawaban:"""