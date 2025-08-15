import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Centralized local regex patterns (no dependency on legal_patterns)
RE_PASAL = re.compile(r"(?mx)^\s*Pasal\s+(?P<pasal_num>\d{1,4})(?P<pasal_suffix>[A-Z])?\s*$")
RE_AYAT = re.compile(r"(?mx)^\s*\(\s*(?P<ayat_num>\d{1,3})\s*\)\s*(?P<ayat_text>.*)$")
RE_HURUF = re.compile(r"(?mx)^\s*(?P<huruf>[a-z])[\.)]\s+(?P<huruf_text>.+)$")
RE_ANGKA = re.compile(r"(?mx)^\s*(?P<angka>\d{1,3})(?P<delim>[.)])\s+(?P<angka_text>.+)$")
RE_BAB = re.compile(r"(?mi)^\s*BAB\s+(?P<bab>[IVXLCDM]+[A-Z]?)\s*$")
RE_PASAL_ROMAWI_AMD = re.compile(r"(?mi)^\s*Pasal\s+[IVXLCDM]+\b")

# Legal language detectors (line-oriented, adapted from comprehensive patterns in perp-1.py)
RE_DEF_LINE = re.compile(r"(?mi)\b(Yang\s+dimaksud\s+dengan|Dalam\s+(?:Undang[‒–-]?Undang|Peraturan)\s+ini\s+yang\s+dimaksud\s+dengan|Untuk\s+keperluan\s+(?:Undang[‒–-]?Undang|Peraturan)\s+ini|Pengertian)\b.*?\b(?:adalah|ialah|yaitu|berarti)\b")
RE_DEF_NUMBERED = re.compile(r"(?m)^\s*\d{1,3}\.[^\n]+\b(?:adalah|yang\s+selanjutnya\s+disebut|yang\s+selanjutnya\s+disingkat)\b.+")

RE_OBLIGATION = re.compile(r"(?mi)\b(wajib|harus|diwajibkan|diharuskan|berkewajiban)\b.*")
RE_PROHIBITION = re.compile(r"(?mi)\b(dilarang|tidak\s+boleh|tidak\s+dapat|tidak\s+diperkenankan|tidak\s+diizinkan)\b.*")
RE_EXCEPTION = re.compile(r"(?mi)\b(kecuali|terkecuali|selain|dengan\s+pengecualian|tidak\s+termasuk|dikecualikan)\b.*")

RE_SANCTION = re.compile(r"(?mi)\b(dipidana|diancam\s+pidana|dikenakan\s+sanksi|dijatuhi\s+sanksi|dapat\s+dipidana)\b.*")
RE_SANCTION_PIDANA = re.compile(r"(?mi)\bpidana\s+(penjara|kurungan|denda)\b")
RE_SANCTION_ADM = re.compile(r"(?mi)\bsanksi\s+administratif\b")
RE_SANCTION_PERDATA = re.compile(r"(?mi)\bsanksi\s+perdata\b")

RE_PROCEDURE = re.compile(r"(?mi)\b(tata\s+cara|prosedur|mekanisme|ketentuan\s+mengenai)\b.*?(?:diatur\s+(?:dalam|dengan|lebih\s+lanjut)|ditetapkan\s+(?:oleh|dengan)|dilaksanakan\s+sesuai\s+dengan).*")

# Optional: basic scope indicators
RE_SCOPE = re.compile(r"(?mi)\b(ruang\s+lingkup|lingkup|cakupan|berlaku\s+untuk|mencakup)\b.*")

# End-of-normative markers
# Penjelasan header can be standalone 'PENJELASAN' or prefixed lines like 'PENJELASAN ATAS ...'
RE_PENJELASAN_HDR = re.compile(r"(?mi)^\s*PENJELASAN(?:\b|[\s:]).*$")
# Common Penjelasan section headings that sometimes appear even if header OCR is noisy
RE_PENJELASAN_I_UMUM = re.compile(r"(?mi)^\s*I\.?\s*UMUM\s*$")
RE_PENJELASAN_PASAL_DEMI = re.compile(r"(?mi)^\s*(?:II\.?\s*)?PASAL\s+DEMI\s+PASAL\s*$")
RE_DITETAPKAN = re.compile(r"(?mi)^\s*Ditetapkan\s+di\b")
RE_DIUNDANGKAN = re.compile(r"(?mi)^\s*Diundangkan\s+di\b")
RE_LEMBARAN_LINE = re.compile(r"(?mi)^\s*LEMBARAN\s+NEGARA\b")
RE_TAMBAHAN_LEMBARAN = re.compile(r"(?mi)^\s*TAMBAHAN\s+LEMBARAN\s+NEGARA\b")
 # Structural unit (roman Pasal used in amendment articles like 'Pasal I, II, ...')
RE_PASAL_ROMAWI_AMD = re.compile(r"(?mi)^\s*Pasal\s+[IVXLCDM]+\b")

# Detect OCR page-break artifacts inside a line
RE_SPACED_DOTS = re.compile(r"(?:\.|·|•)\s*(?:\.|·|•)(?:\s*(?:\.|·|•)){1,}")  # ' . . . ' or bullet variants
RE_MULTI_COMMAS = re.compile(r"(?:,\s*,\s*,\s*,|(?:,\s*){3,})")
RE_ELLIPSIS = re.compile(r"…{1,}")

# Structural higher-level anchors
RE_BAB = re.compile(r"(?mi)^\s*BAB\s+[IVXLCDM]+\s*$")
RE_BAGIAN = re.compile(r"(?mi)^\s*BAGIAN\s+(KE\s*[A-Z]+|[IVXLCDM]+|\d+|[A-Z]+)\s*$")
RE_PARAGRAF = re.compile(r"(?mi)^\s*PARAGRAF\s+([IVXLCDM]+|\d+|[A-Z]+)\s*$")
RE_BUKU = re.compile(r"(?mi)^\s*BUKU\s+([A-Z]+|[IVXLCDM]+|\d+)\s*$")

# Konsiderans & Diktum labels
RE_MENIMBANG_LABEL = re.compile(r"(?mi)^\s*Menimbang\s*:\s*(.*)$")
RE_MENGINGAT_LABEL = re.compile(r"(?mi)^\s*Mengingat\s*:\s*(.*)$")
RE_MEMUTUSKAN_LABEL = re.compile(r"(?mi)^\s*MEMUTUSKAN\s*:\s*(.*)$")
RE_MENETAPKAN_LABEL = re.compile(r"(?mi)^\s*Menetapkan\s*:\s*(.*)$")


def has_pagebreak_artifact(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if RE_SPACED_DOTS.search(s):
        return True
    if RE_MULTI_COMMAS.search(s):
        return True
    if RE_ELLIPSIS.search(s):
        return True
    # Specific common duplicates like "Paragraf 4 . . ." or ", , ,,,"
    if re.search(r"\bParagraf\b\s*\d+\s*(?:[\.|·|•]\s*){2,}$", s):
        return True
    return False


def remove_artifact_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        if has_pagebreak_artifact(ln):
            continue  # drop entire line
        out.append(ln.rstrip())
    return out


def normalize_structure(lines: List[str]) -> List[str]:
    out: List[str] = []
    cur_ctx = {
        "in_pasal": False,
        "pasal_label": None,
        "current_ayat": None,
        "section": None,  # one of: None, 'menimbang', 'mengingat', 'memutuskan'
        "in_preamble": False,  # inside Menimbang/Mengingat/Memutuskan/Menetapkan block
        "root_sep_inserted": False,  # whether '<!-- root document -->' inserted
    }

    def flush_blank_conditional(prev: str, nxt: str) -> bool:
        # Keep single blank between major headers; drop within pasal body unless separating units
        if not prev or not nxt:
            return False
        # Option A (compact):
        # - Keep blank BEFORE a Pasal header (nxt is Pasal)
        # - Do NOT keep blank AFTER a Pasal header (prev is Pasal)
        if RE_PASAL.match(nxt):
            return True
        if RE_PASAL.match(prev):
            return False
        if RE_BAB.match(prev) or RE_BAGIAN.match(prev) or RE_PARAGRAF.match(prev) or RE_BUKU.match(prev):
            return True
        if RE_BAB.match(nxt) or RE_BAGIAN.match(nxt) or RE_PARAGRAF.match(nxt) or RE_BUKU.match(nxt):
            return True
        # Within pasal or within konsiderans/diktum sections: do NOT keep blank lines
        if cur_ctx["in_pasal"] or cur_ctx["section"] in {"menimbang", "mengingat", "memutuskan"}:
            return False
        return False

    i = 0
    N = len(lines)
    while i < N:
        raw = lines[i].rstrip()
        line = raw

        # Collapse repeated spaces
        line = re.sub(r"\s+", " ", line) if not line.startswith(" ") else line

        if not line.strip():
            # blank line: decide to keep or drop based on neighbors
            prev = out[-1] if out else ""
            nxt = lines[i+1].rstrip() if i+1 < N else ""
            if flush_blank_conditional(prev, nxt):
                if prev.strip():
                    out.append("")
            i += 1
            continue

        # Header lines: ensure title on its own line (next non-empty is title, keep as-is)
        if RE_BUKU.match(line) or RE_BAB.match(line) or RE_BAGIAN.match(line) or RE_PARAGRAF.match(line):
            cur_ctx.update({"in_pasal": False, "pasal_label": None, "current_ayat": None, "section": None, "in_preamble": False})
            # Ensure previous is separated
            if out and out[-1] != "":
                out.append("")
            out.append(line.strip())
            # If next line is an ALL CAPS title glued on same line (rare), split – conservative: no-op
            i += 1
            continue

        # Konsiderans & Diktum labels handling
        m_menimbang = RE_MENIMBANG_LABEL.match(line)
        if m_menimbang:
            cur_ctx.update({"in_pasal": False, "pasal_label": None, "current_ayat": None, "section": "menimbang", "in_preamble": True})
            # Normalize label to its own line, ignore any trailing text on same line (will be part of first item if present)
            if out and out[-1] != "":
                out.append("")
            out.append("Menimbang :")
            # If there is trailing content on the same line, push it as continuation of first item
            trailing = m_menimbang.group(1).strip()
            if trailing:
                # If trailing starts like 'a.' or similar, let later huruf logic format it; else keep
                out.append(trailing)
            i += 1
            continue

        m_mengingat = RE_MENGINGAT_LABEL.match(line)
        if m_mengingat:
            cur_ctx.update({"in_pasal": False, "pasal_label": None, "current_ayat": None, "section": "mengingat", "in_preamble": True})
            if out and out[-1] != "":
                out.append("")
            out.append("Mengingat :")
            trailing = m_mengingat.group(1).strip()
            if trailing:
                out.append(trailing)
            i += 1
            continue

        m_memutuskan = RE_MEMUTUSKAN_LABEL.match(line)
        if m_memutuskan:
            cur_ctx.update({"in_pasal": False, "pasal_label": None, "current_ayat": None, "section": "memutuskan", "in_preamble": True})
            if out and out[-1] != "":
                out.append("")
            out.append("MEMUTUSKAN:")
            trailing = m_memutuskan.group(1).strip()
            if trailing:
                out.append(trailing)
            i += 1
            continue

        m_menetapkan = RE_MENETAPKAN_LABEL.match(line)
        if m_menetapkan and cur_ctx["section"] == "memutuskan":
            # Keep Menetapkan on same block, no blank lines
            content = m_menetapkan.group(1).strip()
            if content:
                out.append(f"Menetapkan : {content}")
            else:
                out.append("Menetapkan :")
            i += 1
            continue

        # Pasal header must be its own line
        m_pas = RE_PASAL.match(line)
        if m_pas:
            # Insert root document separator once when transitioning from preamble to first Pasal
            if cur_ctx.get("in_preamble") and not cur_ctx.get("root_sep_inserted"):
                if out and out[-1] != "":
                    out.append("")
                out.append("<!-- root document -->")
                out.append("")
                cur_ctx["root_sep_inserted"] = True
                cur_ctx["in_preamble"] = False
            cur_ctx.update({"in_pasal": True, "pasal_label": f"{m_pas.group('pasal_num')}{m_pas.group('pasal_suffix') or ''}", "current_ayat": None})
            if out and out[-1] != "":
                out.append("")
            out.append(f"Pasal {cur_ctx['pasal_label']}")
            i += 1
            continue

        # Ayat line
        m_ay = RE_AYAT.match(line)
        if m_ay:
            cur_ctx["current_ayat"] = m_ay.group("ayat_num")
            text = m_ay.group("ayat_text").strip()
            out.append(f"({cur_ctx['current_ayat']}) {text}")
            i += 1
            continue

        # Huruf line – attach under last ayat; if no ayat, keep but normalize
        m_h = RE_HURUF.match(line)
        if m_h:
            h = m_h.group("huruf")
            text = m_h.group("huruf_text").strip()
            # ensure no blank line before huruf (compact)
            if out and out[-1] == "":
                out.pop()
            out.append(f"{h}. {text}")
            i += 1
            continue

        # Angka line – prefer nesting under last huruf; if not available, under ayat
        m_k = RE_ANGKA.match(line)
        if m_k:
            k = m_k.group("angka")
            delim = m_k.group("delim") if 'delim' in m_k.re.groupindex else ')'
            text = m_k.group("angka_text").strip()
            out.append(f"{k}{delim} {text}")
            i += 1
            continue

        # Generic text: if inside pasal and previous line is ayat/huruf/angka, append continuation
        if cur_ctx["in_pasal"] and out:
            prev = out[-1]
            if re.match(r"^(\(\d+\)|[a-z][\.)]\s|\d+[.)])", prev):
                # continuation line for wrapped text
                out[-1] = prev.rstrip() + " " + line.strip()
            else:
                out.append(line.strip())
        else:
            out.append(line.strip())
        i += 1

    # Final tidy: remove excessive blank lines (max 2 between major sections, max 1 elsewhere)
    tidy: List[str] = []
    blank_run = 0
    for ln in out:
        if ln.strip():
            tidy.append(ln)
            blank_run = 0
        else:
            if blank_run == 0:
                tidy.append("")
            blank_run = 1
    return tidy


def _infer_anchor(line: str, cur_anchor: dict) -> dict:
    """Update structural anchor context based on the current normalized line.
    cur_anchor keys: pasal, ayat, huruf, angka
    """
    if RE_PASAL.match(line):
        m = RE_PASAL.match(line)
        cur_anchor = {"pasal": f"Pasal {m.group('pasal_num')}{m.group('pasal_suffix') or ''}", "ayat": None, "huruf": None, "angka": None}
        return cur_anchor
    m = RE_AYAT.match(line)
    if m:
        cur_anchor.update({"ayat": f"ayat ({m.group('ayat_num')})", "huruf": None, "angka": None})
        return cur_anchor
    m = RE_HURUF.match(line)
    if m:
        cur_anchor.update({"huruf": f"huruf {m.group('huruf')}", "angka": None})
        return cur_anchor
    m = RE_ANGKA.match(line)
    if m:
        cur_anchor.update({"angka": f"angka {m.group('angka')}"})
        return cur_anchor
    return cur_anchor


def _anchor_str(cur_anchor: dict) -> str:
    parts: List[str] = []
    if cur_anchor.get("pasal"):
        parts.append(cur_anchor["pasal"])
    if cur_anchor.get("ayat"):
        parts.append(cur_anchor["ayat"])
    if cur_anchor.get("huruf"):
        parts.append(cur_anchor["huruf"])
    if cur_anchor.get("angka"):
        parts.append(cur_anchor["angka"])
    return " ".join(parts)


def _classify_sanction(text: str) -> str:
    if RE_SANCTION_ADM.search(text):
        return "sanction_administratif"
    if RE_SANCTION_PERDATA.search(text):
        return "sanction_perdata"
    if RE_SANCTION_PIDANA.search(text):
        return "sanction_pidana"
    return "sanction"


def _find_normative_end(tidy_lines: List[str], start_idx: int) -> int:
    """Return index where normative content ends (first line index of non-normative blocks)."""
    # Prefer legal_patterns 'lembaran_negara' if available by scanning lines
    for i in range(start_idx + 1, len(tidy_lines)):
        ln = tidy_lines[i]
        if (
            RE_PENJELASAN_HDR.match(ln)
            or RE_PENJELASAN_I_UMUM.match(ln)
            or RE_PENJELASAN_PASAL_DEMI.match(ln)
            or RE_DITETAPKAN.match(ln)
            or RE_DIUNDANGKAN.match(ln)
            or RE_LEMBARAN_LINE.match(ln)
            or RE_TAMBAHAN_LEMBARAN.match(ln)
        ):
            return i
    return len(tidy_lines)


def insert_section_markers(tidy_lines: List[str], src_doc: Optional[str]) -> List[str]:
    """Scan tidy normalized lines for legal language units and insert <!-- section --> blocks
    immediately after the '<!-- root document -->' separator.
    """
    # Normalize root markers: strip any existing root open/close, and insert opener at top
    lines_wo_root = [
        ln for ln in tidy_lines if ln.strip() not in ("<!-- root document -->", "<!-- end root document -->")
    ]
    tidy_lines = ["<!-- root document -->", ""] + lines_wo_root
    root_idx = 0

    # Determine normative content start: first structural unit after root (Pasal/BAB/BAGIAN/PARAGRAF)
    start_norm = None
    for i in range(root_idx + 1, len(tidy_lines)):
        ln = tidy_lines[i]
        if (
            RE_PASAL.match(ln)
            or RE_PASAL_ROMAWI_AMD.match(ln)
            or RE_BAB.match(ln)
            or RE_BAGIAN.match(ln)
            or RE_PARAGRAF.match(ln)
        ):
            start_norm = i
            break
    if start_norm is None:
        # No structural units found; place end of root at end of file
        return tidy_lines + ["<!-- end root document -->", ""]

    # Determine normative content end using the boundary detectors
    norm_end = _find_normative_end(tidy_lines, start_idx=start_norm - 1)

    # Scan main body only within normative range (after separator, before penjelasan/lembaran)
    markers: List[Tuple[int, str, str, str]] = []  # (line_index, unit_type, unit_text, anchor_str)
    cur_anchor = {"pasal": None, "ayat": None, "huruf": None, "angka": None}
    for i in range(start_norm, norm_end):
        ln = tidy_lines[i]
        if not ln.strip():
            continue
        # Update anchor context first
        cur_anchor = _infer_anchor(ln, cur_anchor)

        # Apply detectors on the current line (normalized ayat/huruf lines include continuation text)
        unit_type = None
        if RE_DEF_LINE.search(ln) or RE_DEF_NUMBERED.search(ln):
            unit_type = "definition"
        elif RE_OBLIGATION.search(ln):
            unit_type = "obligation"
        elif RE_PROHIBITION.search(ln):
            unit_type = "prohibition"
        elif RE_EXCEPTION.search(ln):
            unit_type = "exception"
        elif RE_SANCTION.search(ln):
            unit_type = _classify_sanction(ln)
        elif RE_PROCEDURE.search(ln):
            unit_type = "procedure"
        elif RE_SCOPE.search(ln):
            unit_type = "scope"

        if unit_type:
            anchor_here = _anchor_str(cur_anchor)
            markers.append((i, unit_type, ln.strip(), anchor_here))

    if not markers:
        return tidy_lines

    # Build marker blocks
    block_lines: List[str] = []
    for _, unit_type, text, anchor_str in markers:
        block_lines.extend([
            "<!-- section -->",
            f"section_src_doc={src_doc or ''}",
            f"section_src_unit_id={anchor_str}",
            f"section_unit_type={unit_type}",
            f"section_unit_text={text}",
            "<!-- end section -->",
            "",
        ])

    # Build amendment content blocks (Pasal I, Pasal II, ...) and nested children
    amd_blocks: List[str] = []
    i = start_norm
    # Helper regex for child starts inside amendment blocks (exclude inner definisi numbering)
    RE_AMD_CHILD_START = re.compile(r"(?m)^\s*\d+[\.)]\s+(Ketentuan|Di\s*antara|Diantara|BAB\b|Bab\b|PASAL\b|Pasal\b|Ayat\b|Huruf\b)")
    # Regex to detect amendment operation types
    RE_AMD_TYPE = re.compile(r"(?i)\b(diubah|disisipkan|dihapus|dicabut)\b")
    # Regex to detect affected unit summary on first line
    RE_AFFECT_UNIT = re.compile(r"(?i)\b(BAB\s+[IVXLCDM]+|Pasal\s+\d+[A-Z]?|ayat\s*\(\s*\d+[a-z]?\s*\)|huruf\s+[a-z])\b")
    # Regex to detect source doc mention in preface (simple heuristic)
    RE_SRC_DOC = re.compile(r"(?i)(Undang-Undang|Peraturan\s+(?:Pemerintah|Presiden))\s+Nomor\s+\d+[A-Z]?(?:\s+Tahun\s+\d{4})?")
    # Detailed operation heuristics
    RE_INSERT_BAB_BETWEEN = re.compile(r"(?i)Di\s*antara\s*BAB\s*([IVXLCDM]+)\s*dan\s*BAB\s*([IVXLCDM]+)\s*disisipkan.*?BAB\s*([IVXLCDM]+[A-Z]?)")
    RE_INSERT_PASAL_BETWEEN = re.compile(r"(?i)Di\s*antara\s*Pasal\s*(\d+[A-Z]?)\s*dan\s*Pasal\s*(\d+[A-Z]?)\s*disisipkan.*?Pasal\s*(\d+[A-Z]?)")
    RE_MODIFY_PASAL = re.compile(r"(?i)Ketentuan\s+(?:angka\s+\d+\s+)?Pasal\s*(\d+[A-Z]?)\s*diubah")
    RE_DELETE_PASAL = re.compile(r"(?i)Ketentuan\s+Pasal\s*(\d+[A-Z]?)\s*dihapus")
    RE_REVOKE_PASAL = re.compile(r"(?i)Ketentuan\s+Pasal\s*(\d+[A-Z]?)\s*dicabut")
    RE_MODIFY_AYAT = re.compile(r"(?i)Ketentuan\s+ayat\s*\(\s*(\d+[a-z]?)\s*\)\s+Pasal\s*(\d+[A-Z]?)\s*diubah")
    RE_INSERT_AYAT = re.compile(r"(?i)di\s*antara\s*ayat\s*\(\s*(\d+[a-z]?)\s*\)\s*dan\s*ayat\s*\(\s*(\d+[a-z]?)\s*\).*?disisipkan\s*ayat\s*\(\s*(\d+[a-z]?)\s*\)")
    RE_MODIFY_HURUF = re.compile(r"(?i)Ketentuan\s+huruf\s*([a-z])\s+Pasal\s*(\d+[A-Z]?)\s*diubah")
    RE_DELETE_HURUF = re.compile(r"(?i)Ketentuan\s+huruf\s*([a-z])\s+Pasal\s*(\d+[A-Z]?)\s*dihapus")

    def _infer_child_meta(header_line: str, child_text: str) -> dict:
        text = header_line + " " + child_text
        text = " ".join(text.split())
        meta = {
            "operation": "",
            "target_unit_type": "",
            "target_unit_id": "",
            "anchor_after": "",
            "anchor_between": "",
        }
        m = RE_INSERT_BAB_BETWEEN.search(text)
        if m:
            a, b, newb = m.groups()
            meta.update({
                "operation": "insert_bab",
                "target_unit_type": "bab",
                "target_unit_id": f"BAB {newb}",
                "anchor_between": f"BAB {a}|BAB {b}",
            })
            return meta
        m = RE_INSERT_PASAL_BETWEEN.search(text)
        if m:
            a, b, newp = m.groups()
            meta.update({
                "operation": "insert_pasal",
                "target_unit_type": "pasal",
                "target_unit_id": f"Pasal {newp}",
                "anchor_between": f"Pasal {a}|Pasal {b}",
            })
            return meta
        m = RE_MODIFY_AYAT.search(text)
        if m:
            ay, ps = m.groups()
            meta.update({
                "operation": "modify_ayat",
                "target_unit_type": "ayat",
                "target_unit_id": f"Pasal {ps} ayat ({ay})",
            })
            return meta
        m = RE_INSERT_AYAT.search(text)
        if m:
            a, b, newa = m.groups()
            meta.update({
                "operation": "insert_ayat",
                "target_unit_type": "ayat",
                "target_unit_id": f"ayat ({newa})",
                "anchor_between": f"ayat ({a})|ayat ({b})",
            })
            return meta
        m = RE_MODIFY_PASAL.search(text)
        if m:
            p = m.group(1)
            meta.update({
                "operation": "modify_pasal",
                "target_unit_type": "pasal",
                "target_unit_id": f"Pasal {p}",
            })
            return meta
        m = RE_DELETE_PASAL.search(text) or RE_REVOKE_PASAL.search(text)
        if m:
            p = m.group(1)
            meta.update({
                "operation": "delete_pasal",
                "target_unit_type": "pasal",
                "target_unit_id": f"Pasal {p}",
            })
            return meta
        m = RE_MODIFY_HURUF.search(text)
        if m:
            h, ps = m.groups()
            meta.update({
                "operation": "modify_huruf",
                "target_unit_type": "huruf",
                "target_unit_id": f"Pasal {ps} huruf {h}",
            })
            return meta
        m = RE_DELETE_HURUF.search(text)
        if m:
            h, ps = m.groups()
            meta.update({
                "operation": "delete_huruf",
                "target_unit_type": "huruf",
                "target_unit_id": f"Pasal {ps} huruf {h}",
            })
            return meta
        # Fallback generic from first matches
        mtype = RE_AMD_TYPE.search(text)
        maff = RE_AFFECT_UNIT.search(header_line)
        meta.update({
            "operation": (mtype.group(1).lower() if mtype else ""),
            "target_unit_type": ("bab" if header_line.lower().startswith("bab ") else ("pasal" if "pasal" in header_line.lower() else "")),
            "target_unit_id": (maff.group(0) if maff else ""),
        })
        return meta

    def _emit_new_units_from_text(lines: List[str]) -> List[str]:
        out: List[str] = []
        for ln in lines:
            if RE_BAB.match(ln):
                out.extend([
                    "<!-- amendment_child_unit -->",
                    "unit_type=bab",
                    f"unit_id={ln.strip()}",
                    f"unit_text={ln.strip()}",
                    "<!-- end amendment_child_unit -->",
                ])
            elif RE_PASAL.match(ln):
                out.extend([
                    "<!-- amendment_child_unit -->",
                    "unit_type=pasal",
                    f"unit_id={ln.strip()}",
                    f"unit_text={ln.strip()}",
                    "<!-- end amendment_child_unit -->",
                ])
            elif RE_AYAT.match(ln):
                m = RE_AYAT.match(ln)
                out.extend([
                    "<!-- amendment_child_unit -->",
                    "unit_type=ayat",
                    f"unit_id=ayat ({m.group('ayat_num')})",
                    f"unit_text={ln.strip()}",
                    "<!-- end amendment_child_unit -->",
                ])
            elif RE_HURUF.match(ln):
                m = RE_HURUF.match(ln)
                out.extend([
                    "<!-- amendment_child_unit -->",
                    "unit_type=huruf",
                    f"unit_id=huruf {m.group('huruf')}",
                    f"unit_text={ln.strip()}",
                    "<!-- end amendment_child_unit -->",
                ])
            elif RE_ANGKA.match(ln):
                m = RE_ANGKA.match(ln)
                out.extend([
                    "<!-- amendment_child_unit -->",
                    "unit_type=angka",
                    f"unit_id=angka {m.group('angka')}",
                    f"unit_text={ln.strip()}",
                    "<!-- end amendment_child_unit -->",
                ])
        return out
    while i < norm_end:
        ln = tidy_lines[i]
        if RE_PASAL_ROMAWI_AMD.match(ln):
            title = ln.strip()
            j = i + 1
            while j < norm_end and not RE_PASAL_ROMAWI_AMD.match(tidy_lines[j]):
                j += 1
            content = tidy_lines[i:j]
            # Extract preface (between title and first child) for src_amd_doc
            # Find child starts
            child_indices: List[int] = []
            for k in range(i + 1, j):
                if RE_AMD_CHILD_START.match(tidy_lines[k] or ""):
                    child_indices.append(k)
            # Build amendment outer block
            amd_blocks.extend([
                "<!-- content_amendment -->",
                f"amendment_title={title}",
            ] + content[:1] + [])
            # Preface lines are from i+1 to first child or j
            preface_end = child_indices[0] if child_indices else j
            preface_text = "\n".join([ln for ln in tidy_lines[i+1:preface_end] if ln.strip()])
            msrc = RE_SRC_DOC.search(preface_text)
            src_amd_doc = msrc.group(0).strip() if msrc else ""
            # Emit children blocks
            if child_indices:
                child_indices.append(j)  # sentinel end
                for ci in range(len(child_indices) - 1):
                    s = child_indices[ci]
                    e = child_indices[ci + 1]
                    child_lines = tidy_lines[s:e]
                    header_line = child_lines[0] if child_lines else ""
                    mtype = RE_AMD_TYPE.search(" ".join(child_lines))
                    amd_type = (mtype.group(1).lower() if mtype else "")
                    maff = RE_AFFECT_UNIT.search(header_line)
                    affected_unit = maff.group(0) if maff else ""
                    meta = _infer_child_meta(header_line, " ".join(child_lines))
                    # Detect new text start after phrase 'sehingga berbunyi' or 'sebagai berikut'
                    new_text_start = None
                    for idx in range(1, len(child_lines)):
                        if re.search(r"(?i)sehingga\s+berbunyi|sebagai\s+berikut\s*:?$", child_lines[idx-1].strip()):
                            new_text_start = idx
                            break
                    # Emit child block
                    amd_blocks.extend([
                        "<!-- amendment children -->",
                        f"src_amd_doc={src_amd_doc}",
                        f"tgt_amd_doc={src_doc or ''}",
                        f"amd_type={amd_type}",
                        f"affected_unit={affected_unit}",
                        f"operation={meta.get('operation','')}",
                        f"target_unit_type={meta.get('target_unit_type','')}",
                        f"target_unit_id={meta.get('target_unit_id','')}",
                    ] + child_lines[:new_text_start if new_text_start is not None else len(child_lines)])
                    if meta.get('anchor_after'):
                        amd_blocks.insert(-1 if new_text_start is not None else len(amd_blocks), f"anchor_after={meta['anchor_after']}")
                    if meta.get('anchor_between'):
                        amd_blocks.insert(-1 if new_text_start is not None else len(amd_blocks), f"anchor_between={meta['anchor_between']}")
                    if new_text_start is not None:
                        amd_blocks.append("<!-- amendment_new_text -->")
                        amd_blocks.extend(child_lines[new_text_start:])
                        # Also emit parsed units for RAG
                        amd_blocks.extend(_emit_new_units_from_text(child_lines[new_text_start:]))
                        amd_blocks.append("<!-- end amendment_new_text -->")
                    amd_blocks.extend(["<!-- end amendment children -->", ""])
            # Close amendment outer block
            amd_blocks.extend([
                "<!-- end content_amendment -->",
                "",
            ])
            i = j
            continue
        i += 1

    # Also include normative content as a consolidated block
    norm_content = tidy_lines[start_norm:norm_end]
    content_block = ["<!-- content -->"] + norm_content + ["<!-- end content -->", ""]

    # Insert end of root before normative start, then sections and content, then remainder
    root_end = ["<!-- end root document -->", ""]
    return tidy_lines[:start_norm] + root_end + block_lines + amd_blocks + content_block + tidy_lines[norm_end:]


def process_text(text: str, src_name: Optional[str] = None) -> str:
    lines = text.splitlines()
    lines = remove_artifact_lines(lines)
    lines = normalize_structure(lines)
    # Insert section markers after normalization
    lines = insert_section_markers(lines, src_name)
    return "\n".join(lines).strip() + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_normalize.py <input_file_or_dir> [--inplace] [--out DIR]")
        sys.exit(1)
    target = Path(sys.argv[1])
    inplace = "--inplace" in sys.argv
    out_dir = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            out_dir = Path(sys.argv[idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if target.is_dir():
        paths = list(target.glob("*.md"))
    else:
        paths = [target]

    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        cleaned = process_text(text, src_name=p.name)
        if inplace and out_dir is None:
            p.write_text(cleaned, encoding="utf-8")
        else:
            dst = (out_dir / p.name) if out_dir else p.with_suffix(".clean.md")
            dst.write_text(cleaned, encoding="utf-8")
            
    print(f"Processed {len(paths)} file(s)")


if __name__ == "__main__":
    main()
