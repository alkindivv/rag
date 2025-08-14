#!/usr/bin/env python3
"""
Jina API Timeout Fix Script
===========================

This script provides immediate fixes for Jina API timeout issues experienced
in the Legal RAG system. It applies optimized timeout configurations and
provides fallback strategies.

Usage:
    python fix_jina_timeout.py                    # Apply recommended fixes
    python fix_jina_timeout.py --test             # Test current configuration
    python fix_jina_timeout.py --aggressive       # Apply aggressive timeout fixes
    python fix_jina_timeout.py --restore          # Restore original settings
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class JinaTimeoutFixer:
    """Fixes Jina API timeout issues with optimized configurations."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.backup_file = self.project_root / ".env.backup"

        # Timeout configurations
        self.configurations = {
            "conservative": {
                "EMBEDDING_REQUEST_TIMEOUT": "60",
                "EMBEDDING_MAX_RETRIES": "2",
                "EMBEDDING_RETRY_DELAY": "2.0",
                "EMBEDDING_BATCH_SIZE": "25",
                "HNSW_EF_SEARCH": "100"
            },
            "balanced": {
                "EMBEDDING_REQUEST_TIMEOUT": "45",
                "EMBEDDING_MAX_RETRIES": "3",
                "EMBEDDING_RETRY_DELAY": "1.5",
                "EMBEDDING_BATCH_SIZE": "30",
                "HNSW_EF_SEARCH": "150"
            },
            "aggressive": {
                "EMBEDDING_REQUEST_TIMEOUT": "120",
                "EMBEDDING_MAX_RETRIES": "1",
                "EMBEDDING_RETRY_DELAY": "5.0",
                "EMBEDDING_BATCH_SIZE": "10",
                "HNSW_EF_SEARCH": "200"
            }
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and level."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "FIX": "🔧"
        }.get(level, "📝")

        print(f"[{timestamp}] {prefix} {message}")

    def backup_env_file(self) -> bool:
        """Create backup of current .env file."""
        try:
            if self.env_file.exists():
                import shutil
                shutil.copy2(self.env_file, self.backup_file)
                self.log(f"Backed up .env to {self.backup_file}", "SUCCESS")
                return True
            else:
                self.log("No .env file found, will create new one", "INFO")
                return True
        except Exception as e:
            self.log(f"Failed to backup .env file: {e}", "ERROR")
            return False

    def read_env_file(self) -> Dict[str, str]:
        """Read current environment variables from .env file."""
        env_vars = {}

        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip().strip('"').strip("'")
            except Exception as e:
                self.log(f"Error reading .env file: {e}", "ERROR")

        return env_vars

    def write_env_file(self, env_vars: Dict[str, str]) -> bool:
        """Write environment variables to .env file."""
        try:
            with open(self.env_file, 'w') as f:
                # Write header
                f.write("# Legal RAG System Configuration\n")
                f.write("# Updated by fix_jina_timeout.py\n")
                f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Group related settings
                groups = {
                    "Database": ["DATABASE_URL"],
                    "Jina API": ["JINA_API_KEY", "JINA_EMBED_BASE", "JINA_EMBED_MODEL"],
                    "Embedding": ["EMBEDDING_MODEL", "EMBEDDING_DIMS", "EMBEDDING_REQUEST_TIMEOUT",
                                "EMBEDDING_MAX_RETRIES", "EMBEDDING_RETRY_DELAY", "EMBEDDING_BATCH_SIZE"],
                    "Search": ["VECTOR_SEARCH_K", "CITATION_CONFIDENCE_THRESHOLD", "HNSW_EF_SEARCH"],
                    "Other": []
                }

                # Write grouped variables
                for group_name, group_keys in groups.items():
                    if group_name != "Other":
                        group_vars = {k: v for k, v in env_vars.items() if k in group_keys}
                        if group_vars:
                            f.write(f"# {group_name}\n")
                            for key, value in group_vars.items():
                                f.write(f"{key}={value}\n")
                            f.write("\n")

                # Write remaining variables
                written_keys = set()
                for group_keys in groups.values():
                    written_keys.update(group_keys)

                remaining_vars = {k: v for k, v in env_vars.items() if k not in written_keys}
                if remaining_vars:
                    f.write("# Other Settings\n")
                    for key, value in remaining_vars.items():
                        f.write(f"{key}={value}\n")

            self.log("Updated .env file with new configuration", "SUCCESS")
            return True

        except Exception as e:
            self.log(f"Failed to write .env file: {e}", "ERROR")
            return False

    def apply_timeout_fixes(self, config_type: str = "balanced") -> bool:
        """Apply timeout fixes with specified configuration."""
        self.log(f"Applying {config_type} timeout configuration...", "FIX")

        # Backup current configuration
        if not self.backup_env_file():
            return False

        # Read current environment
        env_vars = self.read_env_file()

        # Apply new configuration
        config = self.configurations.get(config_type, self.configurations["balanced"])

        for key, value in config.items():
            old_value = env_vars.get(key, "not set")
            env_vars[key] = value
            self.log(f"  {key}: {old_value} → {value}", "FIX")

        # Ensure required variables are set
        required_vars = {
            "JINA_API_KEY": "your_jina_api_key_here",
            "DATABASE_URL": "postgresql://user:pass@localhost/legal_rag",
            "EMBEDDING_DIMS": "384",
            "VECTOR_SEARCH_K": "15"
        }

        for key, default_value in required_vars.items():
            if key not in env_vars or not env_vars[key] or env_vars[key] == "test-key":
                if key == "JINA_API_KEY":
                    self.log(f"⚠️ WARNING: {key} needs to be set manually", "WARNING")
                    env_vars[key] = "your_jina_api_key_here"
                else:
                    env_vars[key] = default_value
                    self.log(f"  {key}: → {default_value} (default)", "FIX")

        # Write updated configuration
        return self.write_env_file(env_vars)

    def test_configuration(self) -> bool:
        """Test current Jina API configuration."""
        self.log("Testing current Jina API configuration...", "INFO")

        try:
            # Try to import and test the diagnostic script
            diagnostic_script = self.project_root / "diagnose_jina_api.py"

            if diagnostic_script.exists():
                self.log("Running quick diagnostic test...", "INFO")
                result = subprocess.run(
                    [sys.executable, str(diagnostic_script), "--quick-test"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    self.log("Diagnostic test PASSED", "SUCCESS")
                    return True
                else:
                    self.log("Diagnostic test FAILED", "ERROR")
                    if result.stdout:
                        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                    if result.stderr:
                        print("STDERR:", result.stderr[-500:])
                    return False
            else:
                self.log("Diagnostic script not found, running basic test", "WARNING")
                return self._basic_api_test()

        except subprocess.TimeoutExpired:
            self.log("Diagnostic test timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error running diagnostic test: {e}", "ERROR")
            return False

    def _basic_api_test(self) -> bool:
        """Run a basic API connectivity test."""
        try:
            # Load environment
            env_vars = self.read_env_file()
            for key, value in env_vars.items():
                os.environ[key] = value

            # Test basic import and initialization
            from src.services.embedding.embedder import JinaV4Embedder
            from src.utils.http import HttpClient

            # Create HTTP client with current timeout settings
            timeout = int(env_vars.get("EMBEDDING_REQUEST_TIMEOUT", "30"))
            max_retries = int(env_vars.get("EMBEDDING_MAX_RETRIES", "3"))

            client = HttpClient(timeout=timeout, max_retries=max_retries)
            embedder = JinaV4Embedder(client=client)

            # Test simple embedding
            self.log(f"Testing embedding with {timeout}s timeout...", "INFO")
            start_time = time.time()

            embedding = embedder.embed_single("test", task="retrieval.query", dims=384)

            duration = (time.time() - start_time) * 1000

            if embedding and len(embedding) == 384:
                self.log(f"Basic API test PASSED ({duration:.1f}ms)", "SUCCESS")
                return True
            else:
                self.log(f"Basic API test FAILED - invalid embedding", "ERROR")
                return False

        except Exception as e:
            self.log(f"Basic API test FAILED: {e}", "ERROR")
            return False

    def restore_backup(self) -> bool:
        """Restore .env file from backup."""
        try:
            if self.backup_file.exists():
                import shutil
                shutil.copy2(self.backup_file, self.env_file)
                self.log("Restored .env from backup", "SUCCESS")
                return True
            else:
                self.log("No backup file found", "WARNING")
                return False
        except Exception as e:
            self.log(f"Failed to restore backup: {e}", "ERROR")
            return False

    def show_current_config(self):
        """Display current timeout configuration."""
        self.log("Current Configuration:", "INFO")

        env_vars = self.read_env_file()

        timeout_keys = [
            "EMBEDDING_REQUEST_TIMEOUT",
            "EMBEDDING_MAX_RETRIES",
            "EMBEDDING_RETRY_DELAY",
            "EMBEDDING_BATCH_SIZE",
            "JINA_API_KEY"
        ]

        for key in timeout_keys:
            value = env_vars.get(key, "not set")
            if key == "JINA_API_KEY" and value and value != "not set":
                # Mask API key
                value = f"{value[:8]}..." if len(value) > 8 else "***"
            self.log(f"  {key}: {value}", "INFO")

    def apply_emergency_fix(self) -> bool:
        """Apply emergency fix for immediate timeout resolution."""
        self.log("🚨 Applying EMERGENCY timeout fix...", "FIX")

        # Emergency configuration - prioritizes working over performance
        emergency_config = {
            "EMBEDDING_REQUEST_TIMEOUT": "180",  # 3 minutes
            "EMBEDDING_MAX_RETRIES": "1",        # Single retry only
            "EMBEDDING_RETRY_DELAY": "10.0",     # Long delay between retries
            "EMBEDDING_BATCH_SIZE": "5",         # Very small batches
            "HNSW_EF_SEARCH": "50"               # Reduced search accuracy for speed
        }

        # Backup and read current config
        if not self.backup_env_file():
            return False

        env_vars = self.read_env_file()

        # Apply emergency settings
        for key, value in emergency_config.items():
            old_value = env_vars.get(key, "not set")
            env_vars[key] = value
            self.log(f"  🚨 {key}: {old_value} → {value}", "FIX")

        # Write configuration
        success = self.write_env_file(env_vars)

        if success:
            self.log("Emergency fix applied! Restart your application.", "SUCCESS")
            self.log("This configuration prioritizes reliability over performance.", "WARNING")
            self.log("Run with --test to verify, then optimize with --balanced", "INFO")

        return success

    def show_recommendations(self):
        """Show recommendations based on common issues."""
        self.log("🔧 TROUBLESHOOTING RECOMMENDATIONS", "INFO")
        self.log("=" * 50, "INFO")

        recommendations = [
            "1. 🔑 Verify API Key: Get free key at https://jina.ai/?sui=apikey",
            "2. 🌐 Check Network: Test internet connectivity and firewall",
            "3. ⏱️ Increase Timeout: Use --conservative for 60s timeout",
            "4. 🔄 Reduce Retries: Fail fast with fewer retry attempts",
            "5. 📦 Smaller Batches: Process fewer texts per request",
            "6. 🚨 Emergency Fix: Use --emergency for immediate resolution",
            "7. 📊 Monitor Status: Check https://status.jina.ai/ for outages",
            "8. 🔍 Run Diagnostics: Use diagnose_jina_api.py for detailed testing"
        ]

        for rec in recommendations:
            self.log(rec, "INFO")

        self.log("", "INFO")
        self.log("Common Environment Variables:", "INFO")
        self.log("  EMBEDDING_REQUEST_TIMEOUT=60    # Timeout in seconds", "INFO")
        self.log("  EMBEDDING_MAX_RETRIES=2         # Number of retries", "INFO")
        self.log("  EMBEDDING_BATCH_SIZE=25         # Texts per request", "INFO")
        self.log("  JINA_API_KEY=your_key_here      # Your Jina API key", "INFO")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix Jina API timeout issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_jina_timeout.py                # Apply balanced fixes
  python fix_jina_timeout.py --conservative # Apply conservative (60s timeout)
  python fix_jina_timeout.py --aggressive   # Apply aggressive (120s timeout)
  python fix_jina_timeout.py --emergency    # Apply emergency fix (180s timeout)
  python fix_jina_timeout.py --test         # Test current configuration
  python fix_jina_timeout.py --restore      # Restore from backup
        """
    )

    parser.add_argument("--conservative", action="store_true",
                       help="Apply conservative timeout settings (60s)")
    parser.add_argument("--aggressive", action="store_true",
                       help="Apply aggressive timeout settings (120s)")
    parser.add_argument("--emergency", action="store_true",
                       help="Apply emergency fix (180s timeout)")
    parser.add_argument("--test", action="store_true",
                       help="Test current configuration")
    parser.add_argument("--restore", action="store_true",
                       help="Restore from backup")
    parser.add_argument("--show-config", action="store_true",
                       help="Show current configuration")

    args = parser.parse_args()

    fixer = JinaTimeoutFixer()

    try:
        if args.restore:
            fixer.restore_backup()

        elif args.show_config:
            fixer.show_current_config()

        elif args.test:
            success = fixer.test_configuration()
            if not success:
                fixer.show_recommendations()
                sys.exit(1)

        elif args.emergency:
            success = fixer.apply_emergency_fix()
            if success:
                fixer.log("Emergency fix applied! Test with: python fix_jina_timeout.py --test", "SUCCESS")
            sys.exit(0 if success else 1)

        elif args.conservative:
            success = fixer.apply_timeout_fixes("conservative")
            if success:
                fixer.log("Conservative fixes applied! Test with: python fix_jina_timeout.py --test", "SUCCESS")
            sys.exit(0 if success else 1)

        elif args.aggressive:
            success = fixer.apply_timeout_fixes("aggressive")
            if success:
                fixer.log("Aggressive fixes applied! Test with: python fix_jina_timeout.py --test", "SUCCESS")
            sys.exit(0 if success else 1)

        else:
            # Default: apply balanced configuration
            fixer.log("Applying balanced timeout configuration...", "INFO")
            success = fixer.apply_timeout_fixes("balanced")

            if success:
                fixer.log("Balanced fixes applied!", "SUCCESS")
                fixer.log("Testing configuration...", "INFO")

                if fixer.test_configuration():
                    fixer.log("🎉 Configuration test PASSED! Jina API should work now.", "SUCCESS")
                else:
                    fixer.log("⚠️ Configuration test FAILED. Try emergency fix:", "WARNING")
                    fixer.log("  python fix_jina_timeout.py --emergency", "WARNING")
                    fixer.show_recommendations()
                    sys.exit(1)
            else:
                fixer.log("Failed to apply fixes", "ERROR")
                sys.exit(1)

    except KeyboardInterrupt:
        fixer.log("Fix process interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        fixer.log(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
