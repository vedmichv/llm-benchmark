#!/usr/bin/env python3
"""
Simple test to verify Ollama is working via CLI
Since WSL can't access Windows localhost directly via HTTP,
we use the ollama CLI which works perfectly.
"""

import subprocess
import sys

def test_ollama_cli():
    """Test Ollama using CLI"""
    print("=" * 60)
    print("Testing Ollama via CLI")
    print("=" * 60)
    print()

    # Test 1: Check ollama version
    print("Test 1: Checking Ollama version...")
    try:
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print()

    # Test 2: List models
    print("Test 2: Listing available models...")
    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"✓ Found {len(lines) - 1} model(s):")
            for line in lines:
                print(f"  {line}")
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print()

    # Test 3: Run a simple prompt
    print("Test 3: Running simple prompt...")
    print("Prompt: 'Count from 1 to 5'")
    try:
        result = subprocess.run(['ollama', 'run', 'qwen3-coder:30b', 'Count from 1 to 5'],
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Clean up ANSI codes
            response = result.stdout.strip()
            # Remove loading spinner characters
            response = ''.join(c for c in response if c.isprintable() or c in '\n\r')
            print(f"✓ Model response:")
            print("-" * 60)
            print(response[-200:] if len(response) > 200 else response)  # Last 200 chars
            print("-" * 60)
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print()
    print("=" * 60)
    print("✓ All tests passed! Ollama is working correctly!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_ollama_cli()
    sys.exit(0 if success else 1)
