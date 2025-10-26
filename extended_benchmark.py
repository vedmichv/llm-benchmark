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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import signal

import ollama
from pydantic import BaseModel, Field, field_validator

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


def save_results_to_markdown(summaries: List[ModelBenchmarkSummary], output_file: str):
    """Save benchmark results to a markdown file"""

    with open(output_file, 'w') as f:
        f.write("# LLM Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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
        "--prompts",
        nargs="*",
        default=[
            # Technical prompts (IT/Infrastructure)
            "Explain the key differences between Kubernetes StatefulSets and Deployments, including when to use each and their specific use cases in production environments.",
            "Compare and contrast DevOps and SRE (Site Reliability Engineering) roles: What are the main responsibilities, skill sets, and philosophies that distinguish these two approaches to managing infrastructure and reliability?",

            # Logic and reasoning
            "You have 12 identical-looking balls. One ball has a different weight (either heavier or lighter than the others, but you don't know which). Using a balance scale exactly 3 times, how can you identify which ball is different and whether it's heavier or lighter? Explain your strategy step by step with clear reasoning.",

            # System design and architecture
            "Design a URL shortening service (like bit.ly) that can handle 1 billion shortened URLs and 10 million requests per day. Explain your architecture, database design, caching strategy, and how you would ensure high availability, reliability, and scalability.",

            # Analytical and comparative thinking
            "Explain why some programming languages are significantly faster than others at runtime. Compare compiled languages (like C++, Rust) versus interpreted languages (like Python, JavaScript), discuss JIT compilation, and explain the trade-offs between development speed and execution speed with real-world examples.",

            # Problem-solving and strategy
            "A remote engineering team of 50 people across 8 different time zones is struggling with communication gaps, delayed decisions, and decreased productivity. Design a comprehensive solution that includes specific tools, processes, meeting structures, and cultural practices to improve team effectiveness and collaboration.",

            # Ethical reasoning and decision-making
            "A company's AI hiring model is 95% accurate overall but has a discovered 2% bias against certain demographic groups in its recommendations. The company has invested 2 years and significant resources into this model. Analyze this situation from multiple perspectives: legal, ethical, business, and technical. What factors should influence their decision, and what would you recommend?"
        ],
        help="Prompts to use for benchmarking"
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

    args = parser.parse_args()

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
    print(f"  Prompts: {len(args.prompts)}")
    print(f"  Runs per prompt: {args.runs_per_prompt}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Output: {args.output}")
    print(f"  Model offloading: {'Disabled' if args.no_offload else 'Enabled'}")

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
    save_results_to_markdown(all_summaries, args.output)

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nBenchmarked {len(all_summaries)} models")
    print(f"Results saved to: {args.output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
