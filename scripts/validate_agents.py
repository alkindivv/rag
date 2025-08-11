#!/usr/bin/env python3
"""
Validator untuk memastikan semua sub-AGENTS.md memiliki struktur yang benar.

Usage:
    python scripts/validate_agents.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple


REQUIRED_SECTIONS = [
    "## Overview",
    "## Scope & Boundaries", 
    "## Inputs & Outputs",
    "## Dependencies",
    "## [PLANNING]",
    "## [EXECUTION]",
    "## [VERIFICATION]",
    "## [TESTS]",
    "## Acceptance Criteria",
    "## Checklist Update Commands"
]

EXPECTED_AGENTS_FILES = [
    "src/services/llm/AGENTS.md",
    "src/services/retriever/AGENTS.md",
    "src/services/search/AGENTS.md",
    "src/services/embedding/AGENTS.md",
    "src/services/answers/AGENTS.md",
    "src/db/AGENTS.md",
    "src/validators/AGENTS.md",
    "src/schemas/AGENTS.md",
    "pipeline/AGENTS.md",
    "tests/AGENTS.md",
    "src/utils/AGENTS.md"
]


def find_repo_root() -> Path:
    """Cari root repository."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists() or (current / "CHECKLIST.md").exists():
            return current
        current = current.parent
    return Path.cwd()


def validate_agent_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Validasi struktur satu file AGENTS.md."""
    errors = []
    
    if not file_path.exists():
        return False, [f"File tidak ditemukan: {file_path}"]
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return False, [f"Gagal membaca file: {e}"]
    
    # Cek required sections
    missing_sections = []
    for section in REQUIRED_SECTIONS:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        errors.append(f"Missing sections: {', '.join(missing_sections)}")
    
    # Cek ada Checklist Update Commands dengan scripts/checklist.py
    if "## Checklist Update Commands" in content:
        checklist_section = content.split("## Checklist Update Commands", 1)
        if len(checklist_section) > 1:
            commands_content = checklist_section[1]
            if "scripts/checklist.py" not in commands_content:
                errors.append("Checklist Update Commands harus menggunakan scripts/checklist.py")
            if "--mark" not in commands_content:
                errors.append("Checklist Update Commands harus memiliki contoh --mark")
    
    # Cek ada Dependencies section dengan anchor references
    if "## Dependencies" in content:
        deps_section = content.split("## Dependencies", 1)
        if len(deps_section) > 1:
            deps_content = deps_section[1].split("##", 1)[0]  # Ambil sampai section berikutnya
            if "anchor:" not in deps_content:
                errors.append("Dependencies section harus menyebutkan anchor checklist")
    
    return len(errors) == 0, errors


def main():
    """Main validation function."""
    repo_root = find_repo_root()
    print(f"Validating AGENTS.md files in: {repo_root}")
    
    all_valid = True
    validation_results = {}
    
    for agent_file in EXPECTED_AGENTS_FILES:
        file_path = repo_root / agent_file
        is_valid, errors = validate_agent_file(file_path)
        validation_results[agent_file] = (is_valid, errors)
        
        if is_valid:
            print(f"✅ {agent_file}")
        else:
            print(f"❌ {agent_file}")
            for error in errors:
                print(f"   - {error}")
            all_valid = False
    
    print("\n" + "="*60)
    
    if all_valid:
        print("🎉 Semua AGENTS.md files valid!")
        return 0
    else:
        failed_count = sum(1 for _, (valid, _) in validation_results.items() if not valid)
        print(f"❌ {failed_count}/{len(EXPECTED_AGENTS_FILES)} files gagal validasi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
