#!/usr/bin/env python3
"""
CLI tool untuk menandai progress di CHECKLIST.md sesuai anchor ID.

Usage:
    python scripts/checklist.py --mark anchor_id
    python scripts/checklist.py --unmark anchor_id
    python scripts/checklist.py --status anchor_id
"""

import argparse
import re
import sys
from pathlib import Path


def find_checklist_file():
    """Cari CHECKLIST.md di root repo."""
    current = Path.cwd()
    while current != current.parent:
        checklist_path = current / "CHECKLIST.md"
        if checklist_path.exists():
            return checklist_path
        current = current.parent
    return None


def update_checklist(anchor_id: str, mark: bool) -> bool:
    """Update status checklist untuk anchor_id tertentu."""
    checklist_path = find_checklist_file()
    if not checklist_path:
        print("ERROR: CHECKLIST.md tidak ditemukan", file=sys.stderr)
        return False
    
    try:
        content = checklist_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Gagal membaca CHECKLIST.md: {e}", file=sys.stderr)
        return False
    
    # Pattern untuk mencari anchor
    if mark:
        # Ubah [ ] menjadi [x]
        pattern = rf'(- \[ \] \(anchor:{re.escape(anchor_id)}\))'
        replacement = rf'- [x] (anchor:{anchor_id})'
    else:
        # Ubah [x] menjadi [ ]
        pattern = rf'(- \[x\] \(anchor:{re.escape(anchor_id)}\))'
        replacement = rf'- [ ] (anchor:{anchor_id})'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print(f"WARNING: Anchor '{anchor_id}' tidak ditemukan atau sudah dalam status yang diminta", file=sys.stderr)
        return False
    
    try:
        checklist_path.write_text(new_content, encoding='utf-8')
        status = "marked" if mark else "unmarked"
        print(f"SUCCESS: Anchor '{anchor_id}' {status}")
        return True
    except Exception as e:
        print(f"ERROR: Gagal menulis CHECKLIST.md: {e}", file=sys.stderr)
        return False


def check_status(anchor_id: str) -> bool:
    """Cek status anchor_id di checklist."""
    checklist_path = find_checklist_file()
    if not checklist_path:
        print("ERROR: CHECKLIST.md tidak ditemukan", file=sys.stderr)
        return False
    
    try:
        content = checklist_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Gagal membaca CHECKLIST.md: {e}", file=sys.stderr)
        return False
    
    # Cari pattern untuk anchor
    marked_pattern = rf'- \[x\] \(anchor:{re.escape(anchor_id)}\)'
    unmarked_pattern = rf'- \[ \] \(anchor:{re.escape(anchor_id)}\)'
    
    if re.search(marked_pattern, content):
        print(f"STATUS: Anchor '{anchor_id}' is MARKED [x]")
        return True
    elif re.search(unmarked_pattern, content):
        print(f"STATUS: Anchor '{anchor_id}' is UNMARKED [ ]")
        return True
    else:
        print(f"ERROR: Anchor '{anchor_id}' tidak ditemukan", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="CLI untuk menandai progress di CHECKLIST.md"
    )
    parser.add_argument("--mark", metavar="ANCHOR_ID", 
                       help="Tandai anchor sebagai selesai [x]")
    parser.add_argument("--unmark", metavar="ANCHOR_ID",
                       help="Tandai anchor sebagai belum selesai [ ]")
    parser.add_argument("--status", metavar="ANCHOR_ID",
                       help="Cek status anchor")
    
    args = parser.parse_args()
    
    if not any([args.mark, args.unmark, args.status]):
        parser.print_help()
        sys.exit(1)
    
    success = True
    
    if args.mark:
        success &= update_checklist(args.mark, mark=True)
    
    if args.unmark:
        success &= update_checklist(args.unmark, mark=False)
    
    if args.status:
        success &= check_status(args.status)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
