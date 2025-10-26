# LLM Benchmark Tool

Comprehensive benchmarking tool for Large Language Models (LLMs) running locally via [Ollama](https://ollama.com/). Measures token throughput (tokens/second), response quality, and model performance across diverse cognitive tasks.

## 🎯 Features

- **Automated Performance Testing**: Benchmark multiple models with configurable prompts
- **Comprehensive Metrics**: Token throughput, response times, load times, and more
- **System Information Collection**: Auto-captures GPU, CPU, RAM, OS details in results
- **Multiple Export Formats**: Markdown, JSON, and CSV outputs for analysis
- **Results Comparison Tool**: Compare benchmarks across runs to track improvements
- **Diverse Test Suite**: 7 prompts testing technical knowledge, logic, system design, ethics, and problem-solving
- **Model Management**: Automatic model offloading between tests for fair comparisons
- **Concurrent Run Protection**: Lock files prevent simultaneous benchmarks
- **Progress Tracking**: Real-time progress indicators and streaming responses
- **Timestamped Results**: All results saved with timestamps to prevent overwriting
- **Markdown Reports**: Beautiful, detailed results in markdown format
- **Timeout Protection**: 10-minute default timeout with smart handling
- **Ollama Log Diagnostics**: Automatic log capture on failures

## 📊 What Gets Tested

The benchmark includes 7 carefully designed prompts that test:

| Category | Test Type | What It Measures |
|----------|-----------|------------------|
| 🔧 Technical Knowledge | Kubernetes, DevOps/SRE | Domain expertise, infrastructure understanding |
| 🎯 Logical Reasoning | 12 Balls Puzzle | Pure logic, algorithmic thinking |
| 🏗️ System Design | URL Shortener | Architecture skills, scalability thinking |
| 💻 Analytical Thinking | Language Performance | Technical depth, comparative analysis |
| 🌍 Problem-Solving | Remote Team Collaboration | Practical solutions, strategic thinking |
| ⚖️ Ethical Reasoning | AI Bias Decision | Multi-perspective analysis, nuanced judgment |

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Ubuntu 20.04+ (or other Linux distributions)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **GPU**: Optional but highly recommended (NVIDIA GPU with CUDA support)

---

## 📦 Installation

### Step 1: Install Ollama

Ollama is required to run LLMs locally. Install it on Ubuntu:

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify installation:**
```bash
ollama --version
```

**Start Ollama service:**
```bash
# Start in background
ollama serve &

# Or use systemd (recommended for persistent service)
sudo systemctl start ollama
sudo systemctl enable ollama  # Start on boot
```

### Step 2: Download Models

Download the specific models used in our benchmarks:

```bash
# Download all test models (this will take time - ~45GB total)
ollama pull qwen3-coder:30b          # 18GB - Fastest model
ollama pull gpt-oss:20b              # 13GB - Most detailed
ollama pull deepseek-r1:8b           # 5.2GB - Balanced performance
ollama pull deepseek-r1:8b-0528-qwen3-q8_0  # 8.9GB - High precision
```

**Quick option (download smallest model only):**
```bash
ollama pull deepseek-r1:8b           # 5.2GB - Good for testing
```

**Verify models are downloaded:**
```bash
ollama list
```

Expected output:
```
NAME                              ID              SIZE      MODIFIED
deepseek-r1:8b-0528-qwen3-q8_0    cade62fd2850    8.9 GB    X minutes ago
deepseek-r1:8b                    6995872bfe4c    5.2 GB    X minutes ago
gpt-oss:20b                       17052f91a42e    13 GB     X minutes ago
qwen3-coder:30b                   06c1097efce0    18 GB     X minutes ago
```

### Step 3: Clone and Setup This Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-benchmark

# Install Python dependencies
pip3 install --break-system-packages -r requirements.txt
# Or use a virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: (Optional) Setup Passwordless Sudo

The benchmark needs sudo to offload models between tests. You have two options:

**Option A: Enter password once per session** (Default)
- Script caches sudo credentials at start
- Prompts for password once, then auto-renews
- No configuration needed

**Option B: Configure passwordless sudo** (Recommended for automation)
```bash
# Run the setup script
./setup_passwordless_sudo.sh

# This creates /etc/sudoers.d/ollama-benchmark
# Allows passwordless: pkill, journalctl, pgrep
```

**Option C: Skip offloading** (Fast but less accurate)
```bash
# Run without model offloading (no sudo needed)
python3 extended_benchmark.py --no-offload
```

### Step 5: Verify Setup

Test that everything is working:

```bash
# Quick test with smallest model
python3 extended_benchmark.py \
  --models "deepseek-r1:8b" \
  --runs-per-prompt 1 \
  --prompts "Hello, how are you?"
```

If you see output with benchmark results, you're ready to go! 🎉

---

## 🎮 Usage

### Basic Benchmark (All Models, All Prompts)

```bash
python3 extended_benchmark.py
```

This runs the full benchmark suite:
- **Models**: All 4 downloaded models
- **Prompts**: All 7 default prompts
- **Runs**: 2 per prompt per model (56 total runs)
- **Output**: `benchmark_results_YYYYMMDD_HHMMSS.md`
- **Duration**: ~60-90 minutes

### Quick Test (Single Model)

```bash
python3 extended_benchmark.py --models "qwen3-coder:30b" --runs-per-prompt 1
```

**Result**: Tests fastest model with 7 prompts (1 run each) in ~5-10 minutes

### Test Specific Models

```bash
# Test only fast models
python3 extended_benchmark.py --models "qwen3-coder:30b" "gpt-oss:20b"

# Skip slow models
python3 extended_benchmark.py --skip-models "deepseek-r1:8b-0528-qwen3-q8_0"
```

### Custom Prompts

```bash
python3 extended_benchmark.py \
  --prompts "Explain Docker containers" "What is Kubernetes?" \
  --runs-per-prompt 2
```

### Export to JSON/CSV

```bash
# Export to JSON for data analysis
python3 extended_benchmark.py --export-json results.json

# Export to CSV for Excel/spreadsheets
python3 extended_benchmark.py --export-csv results.csv

# Export to all formats
python3 extended_benchmark.py \
  --export-json results.json \
  --export-csv results.csv
```

**Exports include:**
- System information (GPU, CPU, RAM, OS)
- All performance metrics
- Individual run details
- Timestamps

### Compare Results

Compare benchmark results from different runs to track improvements:

```bash
# Compare two benchmark runs
python3 compare_results.py \
  benchmark_results_20251026_140000.json \
  benchmark_results_20251026_160000.json \
  --labels "Before" "After"
```

**Comparison shows:**
- Performance differences (absolute & percentage)
- System configuration changes
- Per-model improvements
- Summary statistics

### Advanced Options

```bash
python3 extended_benchmark.py \
  --models "qwen3-coder:30b" \
  --runs-per-prompt 3 \
  --timeout 600 \
  --output my_custom_results.md \
  --no-offload
```

**Options:**
- `--models`: Specific models to test (space-separated)
- `--skip-models`: Models to exclude (space-separated)
- `--prompts`: Custom prompts (space-separated, use quotes)
- `--runs-per-prompt`: Number of times to run each prompt (default: 2)
- `--timeout`: Timeout per run in seconds (default: 600 / 10 min)
- `--output`: Custom output filename (default: timestamped)
- `--no-timestamp`: Disable timestamp in output filename
- `--no-offload`: Skip model offloading (faster but less accurate)
- `--force`: Skip all interactive prompts (for automation)
- `--export-json`: Export results to JSON file
- `--export-csv`: Export results to CSV file

---

## 📖 Understanding Results

### Output Files

After running a benchmark, you'll get:

1. **`benchmark_results_YYYYMMDD_HHMMSS.md`** - Main results file
   - System information (GPU, CPU, RAM, OS)
   - Summary table comparing all models
   - Detailed per-model statistics
   - Individual run breakdowns

2. **JSON/CSV exports** (optional)
   - Machine-readable formats for analysis
   - Complete data with system info
   - Easy import into Excel, Python, R

3. **Log file (if using `tee`)** - Complete execution log
   - Real-time progress
   - Model responses (truncated)
   - Timing information

### Reading the Results

Example output structure:

```markdown
## Summary

| Model | Prompt Eval (t/s) | Response (t/s) | Total (t/s) | Avg Prompt Tokens | Avg Response Tokens |
|-------|-------------------|----------------|-------------|-------------------|---------------------|
| qwen3-coder:30b | 2584 | 157.07 | 162.05 | 42 | 1423 |
| gpt-oss:20b | 7441 | 140.59 | 144.99 | 98 | 3289 |
```

**Key Metrics Explained:**

- **Prompt Eval (t/s)**: How fast the model processes input tokens
- **Response (t/s)**: How fast the model generates output tokens ⭐ **Most important**
- **Total (t/s)**: Overall throughput (prompt + response)
- **Avg Prompt Tokens**: Average input length across tests
- **Avg Response Tokens**: Average output length (indicates verbosity)

**Performance Interpretation:**

| Speed (t/s) | Rating | User Experience |
|-------------|--------|-----------------|
| 150+ | ⚡ Excellent | Nearly instant, smooth streaming |
| 100-150 | ✅ Good | Fast, comfortable reading speed |
| 50-100 | 🟡 Moderate | Noticeable delay but usable |
| < 50 | 🔴 Slow | Significant wait times |

### Example Analysis

From our test results:

**Best Overall**: `qwen3-coder:30b` (157 t/s)
- Fastest response generation
- Concise but complete answers
- Best for: Quick responses, coding tasks

**Most Detailed**: `gpt-oss:20b` (141 t/s, 3289 tokens avg)
- Comprehensive explanations
- Good balance of speed and detail
- Best for: In-depth analysis, learning

**Most Efficient**: `deepseek-r1:8b` (111 t/s)
- Best tokens/parameter ratio
- Lower memory usage
- Best for: Resource-constrained systems

---

## 🔧 Troubleshooting

### Ollama Not Running

**Error**: `Connection refused` or `Could not connect to Ollama`

**Solution**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve &

# Or restart service
sudo systemctl restart ollama
```

### Model Not Found

**Error**: `model 'xxx' not found`

**Solution**:
```bash
# List available models
ollama list

# Download missing model
ollama pull <model-name>
```

### Out of Memory

**Error**: System freezes or OOM killer terminates process

**Solutions**:
1. **Test smaller models first**:
   ```bash
   python3 extended_benchmark.py --models "deepseek-r1:8b"
   ```

2. **Reduce concurrent runs**:
   ```bash
   python3 extended_benchmark.py --runs-per-prompt 1
   ```

3. **Enable model offloading** (default behavior):
   - Automatically unloads models between tests
   - Slower but uses less memory

4. **Close other applications** to free RAM

### Timeout Issues

**Error**: Benchmarks timing out on slow hardware

**Solution**:
```bash
# Increase timeout to 10 minutes
python3 extended_benchmark.py --timeout 600

# Or test with shorter prompts
python3 extended_benchmark.py --prompts "Hello" "What is AI?"
```

### Permission Errors

**Error**: `Permission denied` when installing dependencies

**Solution**:
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or install system-wide (not recommended)
pip3 install --break-system-packages -r requirements.txt
```

---

## 📈 Benchmark Tips

### For Accurate Results

1. **Close other applications**: Free up system resources
2. **Use model offloading**: Enabled by default, don't use `--no-offload`
3. **Run multiple times**: Use `--runs-per-prompt 3` for better averages
4. **Test at consistent times**: Avoid thermal throttling from prolonged use

### For Quick Testing

1. **Single model**: `--models "qwen3-coder:30b"`
2. **One run per prompt**: `--runs-per-prompt 1`
3. **Fewer prompts**: `--prompts "Test prompt 1" "Test prompt 2"`
4. **Disable offloading**: `--no-offload` (if testing single model)

### For Comprehensive Analysis

1. **All models**: Don't specify `--models`
2. **Multiple runs**: `--runs-per-prompt 3`
3. **All prompts**: Use default (or add more with `--prompts`)
4. **Enable offloading**: Default behavior

---

## 🎓 Example Workflows

### Workflow 1: Quick Performance Check

```bash
# Test fastest model with 1 prompt
python3 extended_benchmark.py \
  --models "qwen3-coder:30b" \
  --prompts "Explain Docker containers" \
  --runs-per-prompt 1 \
  --no-offload
```

**Time**: ~30 seconds
**Use case**: Quick sanity check

---

### Workflow 2: Compare Two Models

```bash
# Compare small vs large model
python3 extended_benchmark.py \
  --models "deepseek-r1:8b" "qwen3-coder:30b" \
  --runs-per-prompt 2
```

**Time**: ~15-20 minutes
**Use case**: Deciding which model to use

---

### Workflow 3: Full Comprehensive Benchmark

```bash
# Run complete test suite
python3 extended_benchmark.py > benchmark_full.log 2>&1

# Or with live output and log
python3 extended_benchmark.py 2>&1 | tee benchmark_full.log
```

**Time**: 60-90 minutes
**Use case**: Complete performance analysis, documentation

---

### Workflow 4: Test Custom Use Case

```bash
# Test models for coding tasks
python3 extended_benchmark.py \
  --prompts \
    "Write a Python function to parse JSON" \
    "Explain async/await in JavaScript" \
    "Design a REST API for a todo app" \
  --runs-per-prompt 2
```

**Time**: ~30-40 minutes
**Use case**: Domain-specific evaluation

---

## 📚 Additional Resources

### Ollama Documentation
- [Ollama Official Site](https://ollama.com/)
- [Model Library](https://ollama.com/library)
- [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Benchmarking Best Practices
- Run benchmarks when system is idle
- Monitor GPU/CPU temperatures
- Use consistent prompts for fair comparisons
- Document hardware specifications with results

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- Additional prompt categories
- Support for other LLM backends
- GPU utilization metrics
- Cost/performance analysis
- Web UI for results visualization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for making local LLMs accessible
- Model creators: Qwen, DeepSeek, GPT-OSS teams
- Community contributors and testers

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-benchmark/discussions)

---

**Happy Benchmarking! 🚀**
