#!/usr/bin/env python3
"""
Legal RAG CLI - Advanced Terminal Interface
Interactive CLI untuk sistem legal document RAG dengan fitur mirip Claude Code
"""

import asyncio
import json
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse
from pathlib import Path
import readline
import cmd
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.search.hybrid_search import HybridSearchService
from src.services.llm.legal_llm import LegalLLMService
from src.config.settings import settings

class Colors:
    """Terminal colors"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'

    @classmethod
    def format(cls, text: str, color: str) -> str:
        return f"{color}{text}{cls.RESET}"

class LegalRAGCLI(cmd.Cmd):
    """Advanced CLI untuk Legal RAG System"""

    intro = f"""
{Colors.BOLD}{Colors.CYAN}🗳️  Legal RAG CLI{Colors.RESET}
{Colors.DIM}Interactive terminal untuk pencarian dan analisis dokumen hukum{Colors.RESET}

{Colors.YELLOW}Perintah yang tersedia:{Colors.RESET}
  search <query>        - Cari dokumen hukum
  ask <question>        - Tanya pertanyaan hukum
  explain <pasal>       - Jelaskan pasal tertentu
  compare <pasal1> <pasal2> - Bandingkan dua pasal
  history               - Lihat riwayat query
  config                - Lihat konfigurasi
  help                  - Bantuan lengkap
  exit/quit             - Keluar

{Colors.GREEN}Contoh penggunaan:{Colors.RESET}
  search pasal 121 uu 4/2009
  ask apa hukum penambangan tanpa izin?
  explain pasal 35 uu 4/2009
  compare pasal 121 pasal 122 uu 4/2009
"""

    prompt = f"{Colors.CYAN}legal-rag{Colors.RESET}> "

    def __init__(self):
        super().__init__()
        self.search_service = HybridSearchService()
        self.llm_service = LegalLLMService()
        self.history = []
        self.setup_autocomplete()

    def setup_autocomplete(self):
        """Setup autocomplete untuk perintah"""
        commands = ['search', 'ask', 'explain', 'compare', 'history', 'config', 'help', 'exit', 'quit']
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self.complete)

    def complete(self, text, state):
        """Autocomplete untuk perintah"""
        commands = ['search', 'ask', 'explain', 'compare', 'history', 'config', 'help', 'exit', 'quit']
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None

    def format_search_result(self, result, index: int) -> str:
        """Format hasil pencarian"""
        citation = result.citation_string or 'Tidak ada sitasi'
        score = result.score

        return f"""
{Colors.YELLOW}[{index+1}] {Colors.BOLD}{citation}{Colors.RESET}
{Colors.DIM}Score: {score:.3f}{Colors.RESET}
"""



    def format_answer(self, answer: Dict[str, Any]) -> str:
        """Format jawaban LLM"""
        answer_text = answer.get('answer', '')
        confidence = answer.get('confidence', 0)
        duration = answer.get('duration_ms', 0)
        sources = answer.get('sources', [])

        output = f"""
{Colors.GREEN}💡 Jawaban:{Colors.RESET}
{answer_text}

{Colors.BLUE}📊 Statistik:{Colors.RESET}
• Confidence: {confidence:.2f}
• Duration: {duration:.1f}ms
• Sources: {len(sources)}

{Colors.MAGENTA}📚 Sumber:{Colors.RESET}
"""

        for i, source in enumerate(sources[:3], 1):
            output += self.format_search_result(source, i)

        return output

    def do_search(self, arg):
        """Cari dokumen hukum"""
        if not arg:
            print(f"{Colors.RED}❌ Gunakan: search <query>{Colors.RESET}")
            return

        try:
            print(f"{Colors.YELLOW}🔍 Mencari: {arg}{Colors.RESET}")
            results = asyncio.run(self.search_service.search_async(
                query=arg,
                k=5
            ))

            print(f"{Colors.GREEN}✅ Ditemukan {len(results)} hasil{Colors.RESET}")

            for i, result in enumerate(results[:5]):
                print(self.format_search_result(result, i))

            self.history.append({
                'type': 'search',
                'query': arg,
                'timestamp': datetime.now().isoformat(),
                'results': len(results)
            })

        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

    def do_ask(self, arg):
        """Tanya pertanyaan hukum"""
        if not arg:
            print(f"{Colors.RED}❌ Gunakan: ask <pertanyaan>{Colors.RESET}")
            return

        try:
            print(f"{Colors.CYAN}🤔 Menjawab: {arg}{Colors.RESET}")

            # Search for context using async method for better multi-part query handling
            search_results = asyncio.run(self.search_service.search_async(
                query=arg,
                k=5
            ))

            # Generate answer with LLM (now accepts SearchResult objects directly)
            answer = asyncio.run(self.llm_service.generate_answer(
                query=arg,
                context=search_results,
                temperature=0.3,
                max_tokens=1000
            ))

            print(self.format_answer(answer))

            self.history.append({
                'type': 'ask',
                'query': arg,
                'timestamp': datetime.now().isoformat(),
                'confidence': answer.get('confidence', 0)
            })

        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

    def do_explain(self, arg):
        """Jelaskan pasal tertentu"""
        if not arg:
            print(f"{Colors.RED}❌ Gunakan: explain <pasal> [uu] [tahun]{Colors.RESET}")
            return

        try:
            query = f"jelaskan {arg}"
            self.do_ask(query)

        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

    def do_compare(self, arg):
        """Bandingkan dua pasal"""
        parts = arg.split()
        if len(parts) < 2:
            print(f"{Colors.RED}❌ Gunakan: compare <pasal1> <pasal2> [uu] [tahun]{Colors.RESET}")
            return

        try:
            pasal1, pasal2 = parts[0], parts[1]
            query = f"bandingkan pasal {pasal1} dan pasal {pasal2}"
            self.do_ask(query)

        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

    def do_history(self, arg):
        """Lihat riwayat query"""
        if not self.history:
            print(f"{Colors.YELLOW}📋 Belum ada riwayat{Colors.RESET}")
            return

        print(f"{Colors.CYAN}📋 Riwayat Query:{Colors.RESET}")
        for i, item in enumerate(self.history[-10:], 1):
            timestamp = item['timestamp'][:19]
            print(f"{Colors.DIM}[{i}] {item['type'].upper()} - {timestamp}{Colors.RESET}")
            print(f"  {item['query']}")

    def do_config(self, arg):
        """Lihat konfigurasi"""
        print(f"""
{Colors.CYAN}⚙️  Konfigurasi Sistem:{Colors.RESET}
• Provider: {settings.llm_provider}
• Model: {settings.llm_model}
• Embedding: jina-embeddings-v4
• Reranker: jina-reranker-v2
• Database: PostgreSQL + pgvector
        """)

    def do_help(self, arg):
        """Bantuan lengkap"""
        print(self.intro)

    def do_exit(self, arg):
        """Keluar dari CLI"""
        print(f"{Colors.GREEN}👋 Selamat tinggal!{Colors.RESET}")
        return True

    def do_quit(self, arg):
        """Keluar dari CLI"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Handle Ctrl+D"""
        print()
        return self.do_exit(arg)

    def emptyline(self):
        """Handle empty line"""
        pass

    def default(self, line):
        """Handle unknown commands"""
        print(f"{Colors.RED}❌ Perintah tidak dikenal: {line}{Colors.RESET}")
        print(f"{Colors.YELLOW}Ketik 'help' untuk bantuan{Colors.RESET}")

class QuickCLI:
    """CLI cepat untuk satu perintah"""

    def __init__(self):
        self.search_service = HybridSearchService()
        self.llm_service = LegalLLMService()



    async def search(self, query: str, limit: int = 5):
        """Cari cepat"""
        results = await self.search_service.search_async(
            query=query,
            k=limit
        )

        print(f"🔍 Hasil pencarian untuk: {query}")
        for i, result in enumerate(results):
            print(f"[{i+1}] {result.citation_string or 'No citation'}")

    async def ask(self, question: str):
        """Tanya cepat"""
        search_results = await self.search_service.search_async(
            query=question,
            k=5
        )

        answer = await self.llm_service.generate_answer(
            query=question,
            context=search_results,
            temperature=0.3,
            max_tokens=1000
        )

        print(f"💡 {answer['answer']}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Legal RAG CLI')
    parser.add_argument('--interactive', '-i', action='store_true', help='Mode interaktif')
    parser.add_argument('--search', '-s', type=str, help='Cari dokumen')
    parser.add_argument('--ask', '-a', type=str, help='Tanya pertanyaan')
    parser.add_argument('--explain', '-e', type=str, help='Jelaskan pasal')
    parser.add_argument('--compare', '-c', type=str, help='Bandingkan pasal')

    args = parser.parse_args()

    if args.interactive:
        # Mode interaktif
        cli = LegalRAGCLI()
        try:
            cli.cmdloop()
        except KeyboardInterrupt:
            print(f"\n{Colors.GREEN}👋 Selamat tinggal!{Colors.RESET}")

    elif args.search:
        # Mode cepat - search
        quick_cli = QuickCLI()
        asyncio.run(quick_cli.search(args.search))

    elif args.ask:
        # Mode cepat - ask
        quick_cli = QuickCLI()
        asyncio.run(quick_cli.ask(args.ask))

    elif args.explain:
        # Mode cepat - explain
        quick_cli = QuickCLI()
        question = f"jelaskan {args.explain}"
        asyncio.run(quick_cli.ask(question))

    elif args.compare:
        # Mode cepat - compare
        quick_cli = QuickCLI()
        question = f"bandingkan {args.compare}"
        asyncio.run(quick_cli.ask(question))

    else:
        # Default ke mode interaktif
        print("Memulai Legal RAG CLI...")
        cli = LegalRAGCLI()
        try:
            cli.cmdloop()
        except KeyboardInterrupt:
            print(f"\n{Colors.GREEN}👋 Selamat tinggal!{Colors.RESET}")

if __name__ == "__main__":
    main()
