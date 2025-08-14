#!/usr/bin/env python3
"""
Emergency Jina API Timeout Fix
==============================

Quick fix for Jina API timeout issues in Legal RAG system.
This script applies immediate timeout optimizations to resolve
the "Request timed out after 30.0s" errors.

Usage:
    python emergency_jina_fix.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def log(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "FIX": "🔧"}
    print(f"[{timestamp}] {icons.get(level, '📝')} {message}")

def check_env_file():
    """Check if .env file exists and create backup."""
    env_file = Path(".env")
    backup_file = Path(".env.backup")

    if env_file.exists():
        import shutil
        shutil.copy2(env_file, backup_file)
        log(f"Backed up current .env to .env.backup", "SUCCESS")
        return True
    else:
        log("No .env file found, will create new one", "WARNING")
        return False

def apply_emergency_timeouts():
    """Apply emergency timeout settings."""
    log("🚨 Applying EMERGENCY timeout settings...", "FIX")

    # Emergency configuration - prioritize reliability over speed
    emergency_settings = {
        "EMBEDDING_REQUEST_TIMEOUT": "90",    # 90 second timeout
        "EMBEDDING_MAX_RETRIES": "2",         # Only 2 retries max
        "EMBEDDING_RETRY_DELAY": "5.0",       # 5 second delay between retries
        "EMBEDDING_BATCH_SIZE": "10",         # Small batch size
        "HNSW_EF_SEARCH": "100",             # Reduced search complexity
    }

    # Read existing .env
    env_vars = {}
    env_file = Path(".env")

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

    # Apply emergency settings
    for key, value in emergency_settings.items():
        old_value = env_vars.get(key, "not set")
        env_vars[key] = value
        log(f"  {key}: {old_value} → {value}", "FIX")

    # Ensure required settings exist
    if "JINA_API_KEY" not in env_vars or not env_vars["JINA_API_KEY"] or env_vars["JINA_API_KEY"] == "test-key":
        log("⚠️ JINA_API_KEY not set! You need to add your API key manually", "WARNING")
        env_vars["JINA_API_KEY"] = "your_jina_api_key_here"

    if "DATABASE_URL" not in env_vars:
        env_vars["DATABASE_URL"] = "postgresql://user:pass@localhost/legal_rag"

    if "EMBEDDING_DIM" not in env_vars:
        env_vars["EMBEDDING_DIM"] = "384"

    # Write updated .env file
    with open(env_file, 'w') as f:
        f.write("# Legal RAG Emergency Configuration\n")
        f.write(f"# Applied emergency timeout fix at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    log("Emergency settings applied to .env file", "SUCCESS")

def test_jina_connection():
    """Test basic Jina API connection."""
    log("Testing Jina API connection...", "INFO")

    try:
        # Simple HTTP test
        import requests
        response = requests.get("https://api.jina.ai/", timeout=10)
        log(f"Jina API reachable (status: {response.status_code})", "SUCCESS")
        return True
    except Exception as e:
        log(f"Cannot reach Jina API: {e}", "ERROR")
        return False

def restart_api_server():
    """Restart the API server if running."""
    log("Checking for running API server...", "INFO")

    try:
        # Check if server is running on port 8000
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        log("API server is running. Please restart it to apply new settings.", "WARNING")
        log("Kill the server (Ctrl+C) and run: python -m uvicorn src.api.main:app --reload", "INFO")
    except:
        log("No API server detected on localhost:8000", "INFO")

def main():
    """Main emergency fix procedure."""
    log("🚨 EMERGENCY JINA API TIMEOUT FIX", "ERROR")
    log("=" * 50)

    # Step 1: Check environment
    log("Step 1: Checking environment...", "INFO")
    check_env_file()

    # Step 2: Apply emergency timeout settings
    log("Step 2: Applying emergency timeout settings...", "INFO")
    apply_emergency_timeouts()

    # Step 3: Test connectivity
    log("Step 3: Testing basic connectivity...", "INFO")
    if not test_jina_connection():
        log("⚠️ Network connectivity issues detected", "WARNING")
        log("Check your internet connection and firewall settings", "WARNING")

    # Step 4: Provide restart instructions
    log("Step 4: Restart instructions...", "INFO")
    restart_api_server()

    log("=" * 50)
    log("🎯 EMERGENCY FIX COMPLETE", "SUCCESS")
    log("=" * 50)

    print("\n📋 NEXT STEPS:")
    print("1. 🔑 Set your JINA_API_KEY in .env file (get free key: https://jina.ai/?sui=apikey)")
    print("2. 🔄 Restart your API server: python -m uvicorn src.api.main:app --reload")
    print("3. 🧪 Test search: curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{\"query\":\"definisi badan hukum\"}'")
    print("4. 📊 Monitor logs for timeout errors")

    print("\n⚙️ APPLIED SETTINGS:")
    print("- EMBEDDING_REQUEST_TIMEOUT=90 (was 30)")
    print("- EMBEDDING_MAX_RETRIES=2 (was 4)")
    print("- EMBEDDING_BATCH_SIZE=10 (smaller batches)")

    print("\n🔧 IF STILL FAILING:")
    print("- Increase timeout further: EMBEDDING_REQUEST_TIMEOUT=120")
    print("- Reduce batch size: EMBEDDING_BATCH_SIZE=5")
    print("- Check Jina API status: https://status.jina.ai/")
    print("- Use VPN if behind corporate firewall")

    print("\n✅ Settings saved to .env (backup in .env.backup)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Emergency fix interrupted", "WARNING")
        sys.exit(130)
    except Exception as e:
        log(f"Emergency fix failed: {e}", "ERROR")
        sys.exit(1)
