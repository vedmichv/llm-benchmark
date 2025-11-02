# WSL Setup Guide for LLM Benchmark

## Overview

This guide helps you run the LLM benchmark on Windows using WSL (Windows Subsystem for Linux).

## Quick Start

### 1. Install Ollama on Windows

Download and install from [ollama.com/download](https://ollama.com/download)

### 2. Pull a Model

```bash
# From Windows or WSL
ollama pull qwen3-coder:30b
# or any other model
```

### 3. Run the Test

```bash
cd llm-benchmark
python3 test_ollama.py
```

## Known Issue: WSL Networking

The Python `ollama` library cannot connect to Windows Ollama via HTTP from WSL due to networking limitations. However, the `ollama` CLI works perfectly!

### Solution Options

#### Option 1: Use the CLI Test Script (Recommended)

```bash
python3 test_ollama.py
```

This script uses the `ollama` CLI which works perfectly from WSL.

#### Option 2: Configure Ollama for WSL Access

1. **On Windows**, set environment variable:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
   ```

2. **Restart Ollama** (close and reopen the Ollama app)

3. **In WSL**, set the host:
   ```bash
   export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
   ```

4. **Run the benchmark**:
   ```bash
   python3 run.py --models "qwen3-coder:30b" --prompts "Test" --runs-per-prompt 1 --no-offload --force
   ```

#### Option 3: Run on Native Linux

For the full benchmark experience without networking issues, run on native Linux or use the Windows version directly.

## Verifying Setup

### Test Ollama CLI

```bash
# Check version
ollama --version

# List models
ollama list

# Test a prompt
ollama run qwen3-coder:30b "Hello"
```

### Test Python Connection

```bash
# This will fail on WSL due to networking
python3 -c "import ollama; print(ollama.list())"

# But the CLI test works!
python3 test_ollama.py
```

## Model Information

The `qwen3-coder:30b` model used in testing:
- **Size**: 18 GB
- **Best for**: Code generation, technical tasks
- **Speed**: ~150-157 tokens/second (on test system)

## Troubleshooting

### Ollama Not Found

Make sure Ollama is installed on Windows and in your PATH. The WSL integration should work automatically.

### Model Not Found

Download the model first:
```bash
ollama pull qwen3-coder:30b
```

### Connection Refused

The Python HTTP connection won't work from WSL to Windows localhost. Use Option 2 above or stick with the CLI test script.

## Full Benchmark on Windows

To run the full benchmark suite:

1. **Install Python on Windows** (if not already installed)
2. **Open PowerShell** or CMD
3. **Navigate to the repository**:
   ```powershell
   cd path\to\llm-benchmark
   ```
4. **Run the launcher**:
   ```powershell
   python run.py
   ```

This avoids all WSL networking issues!
