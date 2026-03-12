#!/usr/bin/env python3
"""Convenience launcher. Prefer: python -m llm_benchmark"""
import subprocess
import sys

sys.exit(subprocess.call([sys.executable, "-m", "llm_benchmark"] + sys.argv[1:]))
