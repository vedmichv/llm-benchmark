#!/usr/bin/env python3
"""
Extended LLM Benchmark Script
Runs benchmarks on each model separately with proper model offloading,
progress tracking, and detailed result collection.
"""

import argparse
import subprocess
import sys
import time
import os
import atexit
import platform
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import signal

import ollama
from pydantic import BaseModel, Field, field_validator

# Prompt sets for different test sizes
PROMPT_SETS = {
    "small": [
        # Quick test set (3 prompts)
        "Write a Python function to calculate the factorial of a number",
        "Explain the difference between HTTP and HTTPS",
        "Write a binary search algorithm in Python"
    ],
    "medium": [
        # Standard test set (5 prompts) - original defaults
        "Explain the key differences between Kubernetes StatefulSets and Deployments, including when to use each and their specific use cases in production environments.",
        "Compare and contrast DevOps and SRE (Site Reliability Engineering) roles: What are the main responsibilities, skill sets, and philosophies that distinguish these two approaches to managing infrastructure and reliability?",
        "You have 12 identical-looking balls. One ball has a different weight (either heavier or lighter than the others, but you don't know which). Using a balance scale exactly 3 times, how can you identify which ball is different and whether it's heavier or lighter? Explain your strategy step by step with clear reasoning.",
        "Design a URL shortening service (like bit.ly) that can handle 1 billion shortened URLs and 10 million requests per day. Explain your architecture, database design, caching strategy, and how you would ensure high availability, reliability, and scalability.",
        "Explain why some programming languages are significantly faster than others at runtime. Compare compiled languages (like C++, Rust) versus interpreted languages (like Python, JavaScript), discuss JIT compilation, and explain the trade-offs between development speed and execution speed with real-world examples."
    ],
    "large": [
        # Comprehensive test set (11 prompts)
        "Explain the key differences between Kubernetes StatefulSets and Deployments, including when to use each and their specific use cases in production environments.",
        "Compare and contrast DevOps and SRE (Site Reliability Engineering) roles: What are the main responsibilities, skill sets, and philosophies that distinguish these two approaches to managing infrastructure and reliability?",
        "You have 12 identical-looking balls. One ball has a different weight (either heavier or lighter than the others, but you don't know which). Using a balance scale exactly 3 times, how can you identify which ball is different and whether it's heavier or lighter? Explain your strategy step by step with clear reasoning.",
        "Design a URL shortening service (like bit.ly) that can handle 1 billion shortened URLs and 10 million requests per day. Explain your architecture, database design, caching strategy, and how you would ensure high availability, reliability, and scalability.",
        "Explain why some programming languages are significantly faster than others at runtime. Compare compiled languages (like C++, Rust) versus interpreted languages (like Python, JavaScript), discuss JIT compilation, and explain the trade-offs between development speed and execution speed with real-world examples.",
        "Design a distributed caching system like Redis or Memcached. Explain cache eviction policies (LRU, LFU, FIFO), consistency strategies, sharding approaches, and how you would handle cache invalidation in a microservices architecture.",
        "Explain the CAP theorem in distributed systems. Provide real-world examples of databases that prioritize Consistency+Partition tolerance (CP) versus Availability+Partition tolerance (AP), and explain when you would choose each approach.",
        "Write a Python function to implement a rate limiter using the token bucket algorithm. Explain how it works and why it's better than simple request counting for API rate limiting.",
        "Design a real-time notification system (like Firebase Cloud Messaging or AWS SNS) that can handle millions of concurrent users. Explain your technology choices, message queue design, and how you would ensure message delivery guarantees.",
        "Explain how modern search engines like Elasticsearch work. Cover inverted indexes, relevance scoring (TF-IDF, BM25), sharding, and how they achieve near real-time search across billions of documents.",
        "Design the backend architecture for a social media platform like Twitter. Explain how you would handle the feed generation problem (fan-out on write vs fan-out on read), storage strategy for tweets, and how to scale to millions of active users."
    ]
}

# Lock file management
LOCK_FILE = "/tmp/ollama_benchmark.lock"

def create_lock_file():
    """Create lock file to prevent concurrent benchmark runs"""
    if os.path.exists(LOCK_FILE):
        # Check if the process is still running
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())

            # Check if process exists
            try:
                os.kill(old_pid, 0)  # Signal 0 just checks if process exists
                print(f"\n✗ Error: Another benchmark is already running (PID: {old_pid})")
                print(f"   Lock file: {LOCK_FILE}")
                print(f"\n   If this is incorrect, remove the lock file:")
                print(f"   rm {LOCK_FILE}")
                return False
            except OSError:
                # Process doesn't exist - stale lock file
                print(f"⚠️  Removing stale lock file from PID {old_pid}")
                os.remove(LOCK_FILE)
        except (ValueError, IOError):
            # Invalid lock file
            print(f"⚠️  Removing invalid lock file")
            os.remove(LOCK_FILE)

    # Create new lock file with current PID
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        print(f"✓ Lock file created: {LOCK_FILE}")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not create lock file: {e}")
        return True  # Continue anyway

def remove_lock_file():
    """Remove lock file on exit"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except:
        pass

# Register cleanup
atexit.register(remove_lock_file)

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals - cleanup and exit"""
    print("\n\n⚠️  Benchmark interrupted by user")
    remove_lock_file()
    sys.exit(1)

# Register signal handlers for cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print("\n⚠️  Warning: prompt token count was not provided (likely prompt caching)")
            return 0
        return value


class BenchmarkResult(BaseModel):
    """Stores benchmark results for a single run"""
    model: str
    prompt: str
    prompt_eval_ts: float  # tokens per second
    response_ts: float
    total_ts: float
    prompt_tokens: int
    response_tokens: int
    load_time: float
    prompt_eval_time: float
    response_time: float
    total_time: float
    success: bool = True
    error: Optional[str] = None


class ModelBenchmarkSummary(BaseModel):
    """Summary of all benchmark runs for a model"""
    model: str
    runs: List[BenchmarkResult]
    avg_prompt_eval_ts: float
    avg_response_ts: float
    avg_total_ts: float
    avg_prompt_tokens: float
    avg_response_tokens: float
    avg_load_time: float
    avg_prompt_eval_time: float
    avg_response_time: float
    avg_total_time: float


class SystemInfo(BaseModel):
    """System hardware and software information"""
    hostname: str
    os: str
    os_version: str
    python_version: str
    ollama_version: str
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    gpu_available: bool
    gpu_model: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_driver: Optional[str] = None
    gpu_cuda_version: Optional[str] = None


def collect_system_info() -> SystemInfo:
    """Collect comprehensive system information"""

    # Basic system info
    hostname = platform.node()
    os_name = f"{platform.system()} {platform.release()}"
    os_version = platform.version()
    python_version = platform.python_version()

    # Ollama version
    ollama_version = "Unknown"
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ollama_version = result.stdout.strip().replace("ollama version is ", "").replace("ollama version ", "")
    except:
        pass

    # CPU info
    cpu_model = "Unknown"
    cpu_cores = os.cpu_count() or 0
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
    except:
        cpu_model = platform.processor() or "Unknown"

    # RAM info
    ram_total_gb = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    ram_kb = int(line.split()[1])
                    ram_total_gb = ram_kb / (1024 * 1024)
                    break
    except:
        pass

    # GPU info
    gpu_available = False
    gpu_model = None
    gpu_vram_gb = None
    gpu_driver = None
    gpu_cuda_version = None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_available = True
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                gpu_model = parts[0].strip()
                vram_str = parts[1].strip().split()[0]  # "45000 MiB" -> "45000"
                gpu_vram_gb = float(vram_str) / 1024
                gpu_driver = parts[2].strip()

            # Get CUDA version
            cuda_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if cuda_result.returncode == 0:
                gpu_cuda_version = cuda_result.stdout.strip()
    except:
        pass

    return SystemInfo(
        hostname=hostname,
        os=os_name,
        os_version=os_version,
        python_version=python_version,
        ollama_version=ollama_version,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=round(ram_total_gb, 1),
        gpu_available=gpu_available,
        gpu_model=gpu_model,
        gpu_vram_gb=round(gpu_vram_gb, 1) if gpu_vram_gb else None,
        gpu_driver=gpu_driver,
        gpu_cuda_version=gpu_cuda_version
    )


def log_step(step: int, total: int, message: str):
    """Print a formatted step message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[{timestamp}] Step {step}/{total}: {message}")
    print(f"{'='*70}\n")


def get_ollama_logs(lines: int = 50):
    """Get recent Ollama logs from journalctl or stderr"""
    try:
        # Try journalctl first (systemd)
        result = subprocess.run(
            ["journalctl", "-u", "ollama", "-n", str(lines), "--no-pager"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except:
        pass

    # Try getting logs from ollama process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout:
            pid = result.stdout.strip().split('\n')[0]
            # Try to get stderr/stdout from process
            log_result = subprocess.run(
                ["sudo", "tail", "-n", str(lines), f"/proc/{pid}/fd/2"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if log_result.stdout:
                return log_result.stdout
    except:
        pass

    return "Unable to retrieve Ollama logs"


def check_if_model_generating(model_name: str) -> bool:
    """Check if a model is currently generating output"""
    running = check_running_models()
    for line in running:
        if model_name in line:
            # Model is still loaded, likely generating
            return True
    return False


def check_running_models():
    """Check if any models are currently loaded in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Parse output - if there are more than just header lines, models are running
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        # First line is header, rest are running models
        running_models = lines[1:] if len(lines) > 1 else []
        return running_models
    except Exception as e:
        print(f"⚠️  Error checking running models: {e}")
        return []


def ensure_no_models_running():
    """Ensure no models are currently running before starting benchmark"""
    print("\n🔍 Checking for running models...")
    running = check_running_models()

    if running:
        print(f"⚠️  Found {len(running)} model(s) currently loaded:")
        for model in running:
            print(f"    - {model}")
        print("\n🔄 Unloading all models to ensure clean benchmark...")

        # Unload all models
        try:
            # First try graceful unload - ollama will unload after 5 min idle, we force it
            subprocess.run(["sudo", "pkill", "-f", "ollama serve"], timeout=10)
            time.sleep(2)
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)

            # Verify models are gone
            still_running = check_running_models()
            if still_running:
                print(f"⚠️  Warning: {len(still_running)} model(s) still loaded after unload attempt")
                return False
            else:
                print("✓ All models successfully unloaded")
                return True
        except Exception as e:
            print(f"✗ Error unloading models: {e}")
            return False
    else:
        print("✓ No models currently loaded - ready for clean benchmark")
        return True


def nanosec_to_sec(nanosec):
    """Convert nanoseconds to seconds"""
    return nanosec / 1000000000


def offload_model():
    """Offload all models from Ollama memory with verification"""
    log_step(0, 0, "Offloading models from memory")

    # Check if models are running before offload
    running_before = check_running_models()
    if running_before:
        print(f"  Found {len(running_before)} model(s) to offload")

    try:
        # Stop ollama to unload all models
        result = subprocess.run(
            ["sudo", "pkill", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=30
        )
        time.sleep(2)  # Wait for process to stop

        # Restart ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Wait for service to start

        # Verify models are unloaded
        running_after = check_running_models()
        if running_after:
            print(f"⚠️  Warning: {len(running_after)} model(s) still loaded after offload")
            return False
        else:
            print("✓ Models offloaded successfully - memory cleared")
            return True

    except subprocess.TimeoutExpired:
        print("⚠️  Timeout while offloading models")
        return False
    except Exception as e:
        print(f"⚠️  Error offloading models: {e}")
        # Try alternative method - just verify ollama is responsive
        try:
            subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
            print("✓ Ollama service is responsive")
            return True
        except:
            return False


def test_model_load(model_name: str, timeout_seconds: int = 60) -> bool:
    """
    Test if a model loads correctly by sending a simple prompt
    Returns True if successful, False otherwise
    """
    print(f"  → Testing model load for {model_name}...")

    try:
        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            options={"num_predict": 10}  # Limit response length for quick test
        )

        # Cancel alarm
        signal.alarm(0)

        if response and hasattr(response, 'message'):
            print(f"  ✓ Model loaded successfully")
            print(f"    Response preview: {response.message.content[:50]}...")
            return True
        else:
            print(f"  ✗ Model load failed: No valid response")
            return False

    except TimeoutError:
        signal.alarm(0)
        print(f"  ✗ Model load timeout ({timeout_seconds}s)")
        print(f"  ℹ️  Fetching Ollama logs for diagnostics...")
        logs = get_ollama_logs(20)
        print(f"\n  Recent Ollama logs:\n{logs}\n")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"  ✗ Model load error: {e}")
        print(f"  ℹ️  Fetching Ollama logs for diagnostics...")
        logs = get_ollama_logs(20)
        print(f"\n  Recent Ollama logs:\n{logs}\n")
        return False


def run_single_benchmark(
    model_name: str,
    prompt: str,
    run_number: int,
    timeout_seconds: int = 300
) -> Optional[BenchmarkResult]:
    """
    Run a single benchmark with the given model and prompt
    Returns BenchmarkResult or None if failed
    """
    print(f"\n  → Run #{run_number}")
    print(f"    Prompt: {prompt[:60]}...")

    try:
        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        start_time = time.time()

        # Run the benchmark with streaming to show progress
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        print(f"    Response: ", end="", flush=True)
        last_element = None
        char_count = 0
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            char_count += len(content)
            last_element = chunk

            # Show preview only (first 200 chars)
            if char_count > 200:
                print("...", end="", flush=True)
                # Continue consuming stream but don't print
                for remaining_chunk in stream:
                    last_element = remaining_chunk
                break

        print()  # New line after response

        # Cancel alarm
        signal.alarm(0)

        if not last_element:
            return BenchmarkResult(
                model=model_name,
                prompt=prompt,
                prompt_eval_ts=0, response_ts=0, total_ts=0,
                prompt_tokens=0, response_tokens=0,
                load_time=0, prompt_eval_time=0, response_time=0, total_time=0,
                success=False,
                error="No response received"
            )

        # Convert to dict and validate
        if hasattr(last_element, 'model_dump'):
            last_element = last_element.model_dump()
        elif hasattr(last_element, 'dict'):
            last_element = last_element.dict()

        response = OllamaResponse.model_validate(last_element)

        # Calculate metrics
        prompt_eval_ts = response.prompt_eval_count / nanosec_to_sec(response.prompt_eval_duration) if response.prompt_eval_duration > 0 else 0
        response_ts = response.eval_count / nanosec_to_sec(response.eval_duration) if response.eval_duration > 0 else 0
        total_duration = response.prompt_eval_duration + response.eval_duration
        total_ts = (response.prompt_eval_count + response.eval_count) / nanosec_to_sec(total_duration) if total_duration > 0 else 0

        result = BenchmarkResult(
            model=model_name,
            prompt=prompt,
            prompt_eval_ts=prompt_eval_ts,
            response_ts=response_ts,
            total_ts=total_ts,
            prompt_tokens=response.prompt_eval_count,
            response_tokens=response.eval_count,
            load_time=nanosec_to_sec(response.load_duration),
            prompt_eval_time=nanosec_to_sec(response.prompt_eval_duration),
            response_time=nanosec_to_sec(response.eval_duration),
            total_time=nanosec_to_sec(response.total_duration),
            success=True
        )

        # Print quick stats
        print(f"    ✓ Completed: {result.response_tokens} tokens in {result.total_time:.2f}s ({result.response_ts:.2f} t/s)")

        return result

    except TimeoutError:
        signal.alarm(0)
        print(f"\n    ⏱️  Benchmark timeout ({timeout_seconds}s)")

        # Check if model is still generating
        if check_if_model_generating(model_name):
            print(f"    ℹ️  Model is still loaded and may be generating...")
            print(f"    ⏳ Waiting additional 60s for completion...")

            # Give it 60 more seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                # Try to wait for completion
                time.sleep(60)
                signal.alarm(0)
                print(f"    ℹ️  Extended wait completed, but no response captured")
            except TimeoutError:
                signal.alarm(0)
                print(f"    ✗ Extended timeout - giving up")

        print(f"    ℹ️  Fetching Ollama logs for diagnostics...")
        logs = get_ollama_logs(20)
        print(f"\n    Recent Ollama logs:\n{logs}\n")

        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            prompt_eval_ts=0, response_ts=0, total_ts=0,
            prompt_tokens=0, response_tokens=0,
            load_time=0, prompt_eval_time=0, response_time=0, total_time=0,
            success=False,
            error=f"Timeout after {timeout_seconds}s (model may still be generating)"
        )
    except Exception as e:
        signal.alarm(0)
        print(f"\n    ✗ Benchmark error: {e}")
        return BenchmarkResult(
            model=model_name,
            prompt=prompt,
            prompt_eval_ts=0, response_ts=0, total_ts=0,
            prompt_tokens=0, response_tokens=0,
            load_time=0, prompt_eval_time=0, response_time=0, total_time=0,
            success=False,
            error=str(e)
        )


def benchmark_model(
    model_name: str,
    prompts: List[str],
    runs_per_prompt: int = 2,
    timeout_seconds: int = 300
) -> ModelBenchmarkSummary:
    """
    Benchmark a single model with multiple prompts and runs
    """
    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")

        for run in range(runs_per_prompt):
            result = run_single_benchmark(
                model_name=model_name,
                prompt=prompt,
                run_number=run + 1,
                timeout_seconds=timeout_seconds
            )
            if result:
                all_results.append(result)

    # Calculate averages (only from successful runs)
    successful_runs = [r for r in all_results if r.success]

    if not successful_runs:
        print(f"\n  ✗ All benchmark runs failed for {model_name}")
        return ModelBenchmarkSummary(
            model=model_name,
            runs=all_results,
            avg_prompt_eval_ts=0, avg_response_ts=0, avg_total_ts=0,
            avg_prompt_tokens=0, avg_response_tokens=0,
            avg_load_time=0, avg_prompt_eval_time=0,
            avg_response_time=0, avg_total_time=0
        )

    summary = ModelBenchmarkSummary(
        model=model_name,
        runs=all_results,
        avg_prompt_eval_ts=sum(r.prompt_eval_ts for r in successful_runs) / len(successful_runs),
        avg_response_ts=sum(r.response_ts for r in successful_runs) / len(successful_runs),
        avg_total_ts=sum(r.total_ts for r in successful_runs) / len(successful_runs),
        avg_prompt_tokens=sum(r.prompt_tokens for r in successful_runs) / len(successful_runs),
        avg_response_tokens=sum(r.response_tokens for r in successful_runs) / len(successful_runs),
        avg_load_time=sum(r.load_time for r in successful_runs) / len(successful_runs),
        avg_prompt_eval_time=sum(r.prompt_eval_time for r in successful_runs) / len(successful_runs),
        avg_response_time=sum(r.response_time for r in successful_runs) / len(successful_runs),
        avg_total_time=sum(r.total_time for r in successful_runs) / len(successful_runs)
    )

    return summary


def save_results_to_markdown(summaries: List[ModelBenchmarkSummary], output_file: str, system_info: Optional[SystemInfo] = None):
    """Save benchmark results to a markdown file"""

    with open(output_file, 'w') as f:
        f.write("# LLM Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System information
        if system_info:
            f.write("## System Information\n\n")
            f.write(f"- **Hostname:** {system_info.hostname}\n")
            f.write(f"- **OS:** {system_info.os}\n")
            f.write(f"- **Python:** {system_info.python_version}\n")
            f.write(f"- **Ollama:** {system_info.ollama_version}\n")
            f.write(f"- **CPU:** {system_info.cpu_model} ({system_info.cpu_cores} cores)\n")
            f.write(f"- **RAM:** {system_info.ram_total_gb:.1f} GB\n")

            if system_info.gpu_available and system_info.gpu_model:
                f.write(f"- **GPU:** {system_info.gpu_model}")
                if system_info.gpu_vram_gb:
                    f.write(f" ({system_info.gpu_vram_gb:.1f} GB VRAM)")
                f.write("\n")
                if system_info.gpu_driver:
                    f.write(f"- **GPU Driver:** {system_info.gpu_driver}\n")
                if system_info.gpu_cuda_version:
                    f.write(f"- **CUDA:** {system_info.gpu_cuda_version}\n")
            else:
                f.write(f"- **GPU:** Not available (CPU only)\n")

            f.write("\n")

        f.write("---\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Prompt Eval (t/s) | Response (t/s) | Total (t/s) | Avg Prompt Tokens | Avg Response Tokens |\n")
        f.write("|-------|-------------------|----------------|-------------|-------------------|---------------------|\n")

        for summary in summaries:
            f.write(f"| {summary.model} | {summary.avg_prompt_eval_ts:.2f} | {summary.avg_response_ts:.2f} | {summary.avg_total_ts:.2f} | {summary.avg_prompt_tokens:.0f} | {summary.avg_response_tokens:.0f} |\n")

        f.write("\n---\n\n")

        # Detailed results for each model
        f.write("## Detailed Results\n\n")

        for summary in summaries:
            f.write(f"### {summary.model}\n\n")

            # Average stats
            f.write("**Average Performance:**\n\n")
            f.write(f"```\n")
            f.write(f"Prompt eval: {summary.avg_prompt_eval_ts:.2f} t/s\n")
            f.write(f"Response: {summary.avg_response_ts:.2f} t/s\n")
            f.write(f"Total: {summary.avg_total_ts:.2f} t/s\n\n")
            f.write(f"Stats:\n")
            f.write(f"  Prompt tokens: {summary.avg_prompt_tokens:.0f}\n")
            f.write(f"  Response tokens: {summary.avg_response_tokens:.0f}\n")
            f.write(f"  Model load time: {summary.avg_load_time:.2f}s\n")
            f.write(f"  Prompt eval time: {summary.avg_prompt_eval_time:.2f}s\n")
            f.write(f"  Response time: {summary.avg_response_time:.2f}s\n")
            f.write(f"  Total time: {summary.avg_total_time:.2f}s\n")
            f.write(f"```\n\n")

            # Individual runs
            f.write("**Individual Runs:**\n\n")
            for idx, run in enumerate(summary.runs):
                status = "✓" if run.success else "✗"
                f.write(f"{idx + 1}. {status} Prompt: `{run.prompt[:50]}...`\n")
                if run.success:
                    f.write(f"   - Response: {run.response_ts:.2f} t/s, {run.response_tokens} tokens, {run.total_time:.2f}s\n")
                else:
                    f.write(f"   - Error: {run.error}\n")

            f.write("\n")

    print(f"\n✓ Results saved to: {output_file}")


def save_results_to_json(summaries: List[ModelBenchmarkSummary], output_file: str, system_info: Optional[SystemInfo] = None):
    """Save benchmark results to JSON file"""

    data = {
        "generated": datetime.now().isoformat(),
        "system_info": system_info.model_dump() if system_info else None,
        "models": []
    }

    for summary in summaries:
        model_data = {
            "model": summary.model,
            "averages": {
                "prompt_eval_ts": round(summary.avg_prompt_eval_ts, 2),
                "response_ts": round(summary.avg_response_ts, 2),
                "total_ts": round(summary.avg_total_ts, 2),
                "prompt_tokens": round(summary.avg_prompt_tokens, 0),
                "response_tokens": round(summary.avg_response_tokens, 0),
                "load_time": round(summary.avg_load_time, 2),
                "prompt_eval_time": round(summary.avg_prompt_eval_time, 2),
                "response_time": round(summary.avg_response_time, 2),
                "total_time": round(summary.avg_total_time, 2)
            },
            "runs": []
        }

        for run in summary.runs:
            run_data = {
                "prompt": run.prompt,
                "success": run.success,
                "prompt_eval_ts": round(run.prompt_eval_ts, 2),
                "response_ts": round(run.response_ts, 2),
                "total_ts": round(run.total_ts, 2),
                "prompt_tokens": run.prompt_tokens,
                "response_tokens": run.response_tokens,
                "load_time": round(run.load_time, 2),
                "prompt_eval_time": round(run.prompt_eval_time, 2),
                "response_time": round(run.response_time, 2),
                "total_time": round(run.total_time, 2),
                "error": run.error
            }
            model_data["runs"].append(run_data)

        data["models"].append(model_data)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ JSON results saved to: {output_file}")


def save_results_to_csv(summaries: List[ModelBenchmarkSummary], output_file: str, system_info: Optional[SystemInfo] = None):
    """Save benchmark results to CSV file"""

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header with system info
        if system_info:
            writer.writerow(["System Info"])
            writer.writerow(["Hostname", system_info.hostname])
            writer.writerow(["OS", system_info.os])
            writer.writerow(["CPU", f"{system_info.cpu_model} ({system_info.cpu_cores} cores)"])
            writer.writerow(["RAM", f"{system_info.ram_total_gb:.1f} GB"])
            if system_info.gpu_available and system_info.gpu_model:
                writer.writerow(["GPU", f"{system_info.gpu_model} ({system_info.gpu_vram_gb:.1f} GB)"])
            writer.writerow([])

        # Results header
        writer.writerow(["Benchmark Results"])
        writer.writerow(["Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])

        # Summary table
        writer.writerow(["Model", "Prompt Eval (t/s)", "Response (t/s)", "Total (t/s)", "Avg Prompt Tokens", "Avg Response Tokens"])
        for summary in summaries:
            writer.writerow([
                summary.model,
                f"{summary.avg_prompt_eval_ts:.2f}",
                f"{summary.avg_response_ts:.2f}",
                f"{summary.avg_total_ts:.2f}",
                f"{summary.avg_prompt_tokens:.0f}",
                f"{summary.avg_response_tokens:.0f}"
            ])

        writer.writerow([])

        # Detailed results for each model
        for summary in summaries:
            writer.writerow([f"=== {summary.model} ==="])
            writer.writerow(["Run #", "Prompt", "Success", "Response (t/s)", "Response Tokens", "Total Time (s)", "Error"])

            for idx, run in enumerate(summary.runs):
                # Clean error message for CSV (remove newlines)
                error_msg = ""
                if run.error:
                    # Replace newlines with semicolons, limit length
                    error_msg = run.error.replace('\n', '; ').replace('\r', '')
                    # Take first line or first 100 chars
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."

                writer.writerow([
                    idx + 1,
                    run.prompt[:60] + "..." if len(run.prompt) > 60 else run.prompt,
                    "Yes" if run.success else "No",
                    f"{run.response_ts:.2f}" if run.success else "",
                    run.response_tokens if run.success else "",
                    f"{run.total_time:.2f}" if run.success else "",
                    error_msg
                ])

            writer.writerow([])

    print(f"✓ CSV results saved to: {output_file}")


def get_all_models() -> List[str]:
    """Get list of all downloaded Ollama models"""
    try:
        models = ollama.list().models
        model_names = [model.model for model in models]
        return model_names
    except Exception as e:
        print(f"✗ Error getting model list: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extended benchmark for Ollama models with model offloading and progress tracking"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific models to benchmark (default: all downloaded models)"
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Models to skip"
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Predefined prompt set to use: small (3 prompts), medium (5 prompts), large (11 prompts). Default: medium"
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Custom prompts to use for benchmarking (overrides --prompt-set if specified)"
    )
    parser.add_argument(
        "--runs-per-prompt",
        type=int,
        default=2,
        help="Number of times to run each prompt (default: 2)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each benchmark run (default: 600)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file (default: benchmark_results_TIMESTAMP.md)"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamp in output filename"
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Skip model offloading between benchmarks (faster but may affect results)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip all interactive prompts (continue even if models can't be unloaded)"
    )
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export results to JSON file (e.g., --export-json results.json)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file (e.g., --export-csv results.csv)"
    )

    args = parser.parse_args()

    # Select prompts based on prompt-set or custom prompts
    if args.prompts is None:
        # Use predefined prompt set
        prompts = PROMPT_SETS[args.prompt_set]
        print(f"\n📝 Using '{args.prompt_set}' prompt set ({len(prompts)} prompts)")
    else:
        # Use custom prompts
        prompts = args.prompts
        print(f"\n📝 Using {len(prompts)} custom prompts")

    # Update args.prompts for consistency
    args.prompts = prompts

    # Create lock file to prevent concurrent runs
    print("\n🔒 Checking for concurrent benchmark runs...")
    if not create_lock_file():
        return 1

    # Cache sudo credentials at the start (if model offloading is enabled)
    if not args.no_offload:
        print("\n🔑 Caching sudo credentials for model offloading...")
        try:
            result = subprocess.run(
                ["sudo", "-v"],
                timeout=30
            )
            if result.returncode == 0:
                print("✓ Sudo credentials cached")
                # Keep sudo alive in background
                subprocess.Popen(
                    ["bash", "-c", "while true; do sudo -v; sleep 60; done"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                print("⚠️  Warning: Could not cache sudo credentials")
                print("    You may be prompted for password during model offloading")
        except subprocess.TimeoutExpired:
            print("⚠️  Sudo credential timeout - continuing without cached credentials")
        except KeyboardInterrupt:
            print("\n✗ Cancelled by user")
            return 1
        except Exception as e:
            print(f"⚠️  Could not cache sudo: {e}")

    # Generate timestamped output filename if not specified
    if args.output is None:
        if args.no_timestamp:
            args.output = "benchmark_results.md"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"benchmark_results_{timestamp}.md"

    print("\n" + "="*70)
    print("EXTENDED LLM BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    if args.prompts and len(args.prompts) > 0:
        # Check if using predefined set
        prompt_set_name = None
        for set_name, set_prompts in PROMPT_SETS.items():
            if args.prompts == set_prompts:
                prompt_set_name = set_name
                break
        if prompt_set_name:
            print(f"  Prompt set: {prompt_set_name} ({len(args.prompts)} prompts)")
        else:
            print(f"  Prompts: {len(args.prompts)} (custom)")
    else:
        print(f"  Prompts: {len(args.prompts)}")
    print(f"  Runs per prompt: {args.runs_per_prompt}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Output: {args.output}")
    print(f"  Model offloading: {'Disabled' if args.no_offload else 'Enabled'}")

    # Collect system information
    print("\n📊 Collecting system information...")
    system_info = collect_system_info()
    print(f"  CPU: {system_info.cpu_model} ({system_info.cpu_cores} cores)")
    print(f"  RAM: {system_info.ram_total_gb:.1f} GB")
    if system_info.gpu_available and system_info.gpu_model:
        print(f"  GPU: {system_info.gpu_model} ({system_info.gpu_vram_gb:.1f} GB VRAM)")
    else:
        print(f"  GPU: Not available")

    # Get models
    if args.models:
        model_names = args.models
    else:
        model_names = get_all_models()

    # Filter out skipped models
    if args.skip_models:
        model_names = [m for m in model_names if m not in args.skip_models]

    if not model_names:
        print("\n✗ No models to benchmark!")
        return 1

    print(f"\nModels to benchmark: {model_names}")

    # Ensure no models are running before starting benchmark
    if not ensure_no_models_running():
        print("\n⚠️  Warning: Could not verify all models are unloaded")
        print("    Benchmark results may not reflect max capacity")
        if not args.force:
            try:
                response = input("\n    Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("\n✗ Benchmark cancelled")
                    return 1
            except (EOFError, KeyboardInterrupt):
                print("\n✗ Benchmark cancelled")
                return 1
        else:
            print("    --force flag set: Continuing anyway...")

    total_steps = len(model_names) * (2 if not args.no_offload else 1)
    current_step = 0

    all_summaries = []

    # Benchmark each model
    for model_idx, model_name in enumerate(model_names):
        current_step += 1

        # Offload previous model
        if not args.no_offload and model_idx > 0:
            log_step(current_step, total_steps, f"Offloading previous model")
            offload_model()
            current_step += 1

        log_step(current_step, total_steps, f"Benchmarking {model_name} ({model_idx + 1}/{len(model_names)})")

        # Test model load
        if not test_model_load(model_name, timeout_seconds=args.timeout // 5):
            print(f"\n⚠️  Skipping {model_name} - model failed to load\n")
            continue

        # Run benchmarks
        print(f"\n  Running {args.runs_per_prompt} runs for each of {len(args.prompts)} prompts...")
        summary = benchmark_model(
            model_name=model_name,
            prompts=args.prompts,
            runs_per_prompt=args.runs_per_prompt,
            timeout_seconds=args.timeout
        )

        all_summaries.append(summary)

        # Print summary for this model
        print(f"\n  {'─'*60}")
        print(f"  Summary for {model_name}:")
        print(f"    Avg Response: {summary.avg_response_ts:.2f} t/s")
        print(f"    Avg Total: {summary.avg_total_ts:.2f} t/s")
        print(f"    Avg Response Tokens: {summary.avg_response_tokens:.0f}")
        print(f"    Avg Total Time: {summary.avg_total_time:.2f}s")
        print(f"  {'─'*60}")

    # Save results
    print("\n" + "="*70)
    log_step(0, 0, "Saving results")
    save_results_to_markdown(all_summaries, args.output, system_info)

    # Export to JSON if requested
    if args.export_json:
        save_results_to_json(all_summaries, args.export_json, system_info)

    # Export to CSV if requested
    if args.export_csv:
        save_results_to_csv(all_summaries, args.export_csv, system_info)

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nBenchmarked {len(all_summaries)} models")
    print(f"Results saved to: {args.output}")
    if args.export_json:
        print(f"JSON export: {args.export_json}")
    if args.export_csv:
        print(f"CSV export: {args.export_csv}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
