# Windows Setup Guide for LLM Benchmark

Complete guide for running LLM benchmarks on Windows with WSL (Windows Subsystem for Linux).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configure Ollama for WSL Access](#configure-ollama-for-wsl-access)
4. [Running the Benchmark](#running-the-benchmark)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Windows 10/11** (version 2004 or higher)
- **WSL2** (Windows Subsystem for Linux 2)
- **Python 3.8+** (in WSL)
- **Ollama** (installed on Windows)

### Check Your Setup

**1. Verify WSL is installed:**
```powershell
wsl --version
```

If WSL is not installed, run:
```powershell
wsl --install
```

**2. Verify Python in WSL:**
```bash
python3 --version
```

---

## Installation Steps

### Step 1: Install Ollama on Windows

1. Download Ollama from [ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Ollama will start automatically

**Verify Installation:**
```powershell
ollama --version
```

### Step 2: Download AI Models

From PowerShell or CMD:
```powershell
# Download your desired models
ollama pull qwen3-coder:30b
ollama pull deepseek-r1:8b
ollama pull gpt-oss:20b
```

**List installed models:**
```powershell
ollama list
```

### Step 3: Clone the Benchmark Repository

From WSL terminal:
```bash
cd /tmp
git clone https://github.com/vedmichv/llm-benchmark.git
cd llm-benchmark
```

---

## Configure Ollama for WSL Access

By default, Ollama on Windows only listens on `localhost` (127.0.0.1), which WSL cannot access due to networking limitations. Follow these steps to enable WSL access:

### Step 1: Configure Ollama to Listen on All Interfaces

**Open PowerShell as Administrator:**
- Press `Win + X`
- Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

**Run these commands:**
```powershell
# Set Ollama to listen on all interfaces (0.0.0.0)
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')

# Verify the variable was set
[System.Environment]::GetEnvironmentVariable('OLLAMA_HOST', 'User')
```

**Expected output:**
```
0.0.0.0:11434
```

### Step 2: Restart Ollama

**In PowerShell (Admin):**
```powershell
# Stop Ollama
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue

# Wait a moment
Start-Sleep -Seconds 2

# Start Ollama again
Start-Process "ollama" -ArgumentList "serve"
```

### Step 3: Configure Windows Firewall

**In PowerShell (Admin):**
```powershell
# Add firewall rule to allow WSL to access Ollama
New-NetFirewallRule -DisplayName "Ollama WSL Access" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```

**Verify Firewall Rule:**
```powershell
Get-NetFirewallRule -DisplayName "Ollama WSL Access"
```

### Step 4: Configure WSL Environment

**Switch to your WSL terminal and run:**
```bash
# Get Windows host IP and set OLLAMA_HOST
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"

# Verify it's set correctly
echo $OLLAMA_HOST
```

**Expected output:**
```
http://172.21.112.1:11434
```
(Your IP address may be different)

**Make it permanent:**
```bash
# Add to .bashrc so it persists across sessions
echo 'export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '"'"'{ print $3}'"'"'):11434"' >> ~/.bashrc

# Reload .bashrc
source ~/.bashrc
```

### Step 5: Test the Connection

**In WSL, test Python connection:**
```bash
cd /tmp/llm-benchmark
python3 -c "import ollama; print('Testing...'); models = ollama.list(); print(f'✓ Connected! Found {len(models[\"models\"])} model(s)')"
```

**Expected output:**
```
Testing...
✓ Connected! Found 3 model(s)
```

---

## Running the Benchmark

### Option 1: Quick Start (Recommended)

**Single command to run everything:**
```bash
cd /tmp/llm-benchmark
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
python3 run.py
```

This will:
- Check dependencies
- Create virtual environment
- Install required packages
- Run the benchmark

### Option 2: Run Specific Models

**Benchmark specific models:**
```bash
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
cd /tmp/llm-benchmark
python3 extended_benchmark.py --models "qwen3-coder:30b" "deepseek-r1:8b" --prompts "Write a Python function" --runs-per-prompt 3
```

### Option 3: Custom Configuration

**Full control over benchmark parameters:**
```bash
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
cd /tmp/llm-benchmark
python3 extended_benchmark.py \
  --models "qwen3-coder:30b" "deepseek-r1:8b" "gpt-oss:20b" \
  --prompts "Write a Python function to calculate factorial" \
            "Write a function to reverse a string" \
            "Create a binary search implementation" \
  --runs-per-prompt 2 \
  --no-offload \
  --force
```

**Available Options:**
- `--models` - List of models to benchmark (space-separated)
- `--prompts` - List of prompts to test (each in quotes)
- `--runs-per-prompt` - Number of runs per prompt (default: 3)
- `--timeout` - Timeout in seconds per run (default: 600)
- `--no-offload` - Don't attempt to offload models between tests
- `--force` - Continue even if models are already loaded

### View Results

Results are saved to timestamped markdown files:

```bash
# List result files
ls -lh benchmark_results_*.md

# View latest results
cat benchmark_results_*.md | tail -100
```

**Windows Path to Results:**
```
\\wsl.localhost\Ubuntu\tmp\llm-benchmark\benchmark_results_<timestamp>.md
```

You can open this path in Windows Explorer or Notepad.

---

## Troubleshooting

### Issue 1: "Failed to connect to Ollama"

**Symptoms:**
```
Error: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible.
```

**Solutions:**

1. **Check Ollama is running:**
   ```powershell
   # In PowerShell
   Get-Process ollama
   ```

2. **Verify OLLAMA_HOST is set:**
   ```bash
   # In WSL
   echo $OLLAMA_HOST
   ```
   Should output something like `http://172.21.112.1:11434`

3. **Test connection manually:**
   ```bash
   # In WSL
   curl -v http://$(ip route show | grep -i default | awk '{ print $3}'):11434/api/version
   ```

4. **Restart Ollama:**
   ```powershell
   # In PowerShell (Admin)
   Stop-Process -Name "ollama" -Force
   Start-Sleep -Seconds 2
   Start-Process "ollama" -ArgumentList "serve"
   ```

### Issue 2: "Connection timeout"

**Possible Causes:**
- Windows Firewall blocking connection
- OLLAMA_HOST environment variable not set correctly
- Ollama not configured to listen on 0.0.0.0

**Solutions:**

1. **Verify firewall rule exists:**
   ```powershell
   Get-NetFirewallRule -DisplayName "Ollama WSL Access"
   ```

2. **Check Ollama environment variable:**
   ```powershell
   [System.Environment]::GetEnvironmentVariable('OLLAMA_HOST', 'User')
   ```
   Should output: `0.0.0.0:11434`

3. **Temporarily disable Windows Firewall** (for testing only):
   ```powershell
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
   ```

### Issue 3: "python3-venv not available"

**Symptoms:**
```
Error: ensurepip is not available
```

**Solution:**
```bash
# In WSL
sudo apt update
sudo apt install -y python3-venv python3-pip
```

### Issue 4: "externally-managed-environment"

**Symptoms:**
```
error: externally-managed-environment
```

**Solutions:**

1. **Use virtual environment** (recommended):
   ```bash
   cd /tmp/llm-benchmark
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install with --break-system-packages** (alternative):
   ```bash
   pip install --break-system-packages ollama pydantic
   ```

### Issue 5: Permission Denied Errors

**Symptoms:**
```
sudo: a terminal is required to read the password
```

**Solution:**
The benchmark will automatically skip operations requiring sudo. This is normal and won't affect results.

### Issue 6: Finding Results Files

**From Windows:**
1. Open File Explorer
2. In the address bar, type: `\\wsl.localhost\Ubuntu\tmp\llm-benchmark`
3. Look for files named `benchmark_results_<timestamp>.md`

**From WSL:**
```bash
cd /tmp/llm-benchmark
ls -lh benchmark_results_*.md
```

---

## Performance Expectations

Expected performance on your hardware (AMD Ryzen AI 9 HX 370, 24 cores, 15.4 GB RAM):

### qwen3-coder:30b (18 GB model)
- **Response Speed:** ~22 tokens/second
- **Prompt Eval:** ~260 tokens/second
- **Model Load Time:** ~0.1 seconds

### deepseek-r1:8b (5.2 GB model)
- **Response Speed:** ~30-40 tokens/second (expected)
- **Prompt Eval:** ~300+ tokens/second (expected)
- **Model Load Time:** <0.1 seconds

### gpt-oss:20b (13.8 GB model)
- **Response Speed:** ~25-30 tokens/second (expected)
- **Prompt Eval:** ~270+ tokens/second (expected)
- **Model Load Time:** ~0.1 seconds

---

## Advanced Configuration

### Running Multiple Models Sequentially

```bash
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
cd /tmp/llm-benchmark

# Benchmark each model separately
for model in "qwen3-coder:30b" "deepseek-r1:8b" "gpt-oss:20b"; do
  echo "Benchmarking $model..."
  python3 extended_benchmark.py --models "$model" --prompts "Test prompt" --runs-per-prompt 3
done
```

### Comparing Results

```bash
# Use the comparison tool
python3 compare_results.py benchmark_results_*.md
```

### Custom Prompts from File

Create a file with prompts:
```bash
cat > my_prompts.txt << 'EOF'
Write a Python function to calculate factorial
Implement a binary search tree
Create a REST API endpoint
Optimize this SQL query
EOF

# Run benchmark with custom prompts
python3 extended_benchmark.py \
  --models "qwen3-coder:30b" \
  --prompts $(cat my_prompts.txt) \
  --runs-per-prompt 2
```

---

## Quick Reference

### Essential Commands

```bash
# Set environment
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"

# Test connection
python3 -c "import ollama; print(ollama.list())"

# Run benchmark
cd /tmp/llm-benchmark
python3 run.py

# View results
cat benchmark_results_*.md | tail -50
```

### Windows Commands (PowerShell Admin)

```powershell
# Configure Ollama
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')

# Restart Ollama
Stop-Process -Name "ollama" -Force; Start-Sleep -Seconds 2; Start-Process "ollama" -ArgumentList "serve"

# Add firewall rule
New-NetFirewallRule -DisplayName "Ollama WSL Access" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```

---

## Support

For issues or questions:
- **GitHub Issues:** https://github.com/vedmichv/llm-benchmark/issues
- **Ollama Documentation:** https://ollama.com/docs
- **WSL Documentation:** https://learn.microsoft.com/en-us/windows/wsl/

---

## Summary Checklist

Before running benchmarks, ensure:

- [ ] WSL2 is installed and working
- [ ] Python 3.8+ is installed in WSL
- [ ] Ollama is installed on Windows
- [ ] Models are downloaded (ollama pull)
- [ ] OLLAMA_HOST environment variable is set to 0.0.0.0:11434
- [ ] Windows Firewall rule is created for port 11434
- [ ] OLLAMA_HOST is set in WSL to Windows host IP
- [ ] Connection test passes (ollama.list() works)
- [ ] Repository is cloned to /tmp/llm-benchmark

**Once all checkboxes are complete, run:**
```bash
cd /tmp/llm-benchmark
export OLLAMA_HOST="http://$(ip route show | grep -i default | awk '{ print $3}'):11434"
python3 run.py
```

**Results will be saved to:** `benchmark_results_<timestamp>.md`

---

*Last updated: November 2, 2025*
