#!/usr/bin/env python3
"""
LLM Benchmark Results Comparison Tool
Compares multiple benchmark results to track improvements and differences.
"""

import argparse
import json
import sys
from typing import List, Dict, Any
from pathlib import Path


def load_json_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def compare_results(results_list: List[Dict[str, Any]], labels: List[str]):
    """Compare multiple benchmark results and display differences"""

    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    # Print system info comparison if available
    print("\n## System Information\n")
    for idx, (results, label) in enumerate(zip(results_list, labels)):
        sys_info = results.get("system_info")
        if sys_info:
            print(f"### {label}")
            print(f"  - GPU: {sys_info.get('gpu_model', 'N/A')} ({sys_info.get('gpu_vram_gb', 0):.1f} GB)")
            print(f"  - CPU: {sys_info.get('cpu_model', 'N/A')} ({sys_info.get('cpu_cores', 0)} cores)")
            print(f"  - RAM: {sys_info.get('ram_total_gb', 0):.1f} GB")
            print(f"  - Ollama: {sys_info.get('ollama_version', 'N/A')}")
        else:
            print(f"### {label}")
            print(f"  - System info not available")
        print()

    # Get all unique models across all results
    all_models = set()
    for results in results_list:
        for model in results.get("models", []):
            all_models.add(model["model"])

    # Compare each model
    print("="*80)
    print("\n## Model Performance Comparison\n")

    for model_name in sorted(all_models):
        print(f"### {model_name}\n")

        # Create comparison table header
        header = f"{'Metric':<25}"
        for label in labels:
            header += f" | {label:<15}"
        header += " | Difference"
        print(header)
        print("-" * len(header))

        # Collect stats for this model from each result
        model_stats = []
        for results in results_list:
            model_data = next((m for m in results.get("models", []) if m["model"] == model_name), None)
            if model_data:
                model_stats.append(model_data["averages"])
            else:
                model_stats.append(None)

        # Response t/s
        values = [stats["response_ts"] if stats else None for stats in model_stats]
        row = f"{'Response (t/s)':<25}"
        for val in values:
            row += f" | {val:>15.2f}" if val is not None else f" | {'N/A':>15}"

        if all(v is not None for v in values) and len(values) >= 2:
            diff = values[-1] - values[0]
            pct_change = (diff / values[0] * 100) if values[0] != 0 else 0
            row += f" | {diff:+.2f} ({pct_change:+.1f}%)"
        else:
            row += f" | -"
        print(row)

        # Total t/s
        values = [stats["total_ts"] if stats else None for stats in model_stats]
        row = f"{'Total (t/s)':<25}"
        for val in values:
            row += f" | {val:>15.2f}" if val is not None else f" | {'N/A':>15}"

        if all(v is not None for v in values) and len(values) >= 2:
            diff = values[-1] - values[0]
            pct_change = (diff / values[0] * 100) if values[0] != 0 else 0
            row += f" | {diff:+.2f} ({pct_change:+.1f}%)"
        else:
            row += f" | -"
        print(row)

        # Response tokens
        values = [stats["response_tokens"] if stats else None for stats in model_stats]
        row = f"{'Response Tokens':<25}"
        for val in values:
            row += f" | {val:>15.0f}" if val is not None else f" | {'N/A':>15}"

        if all(v is not None for v in values) and len(values) >= 2:
            diff = values[-1] - values[0]
            pct_change = (diff / values[0] * 100) if values[0] != 0 else 0
            row += f" | {diff:+.0f} ({pct_change:+.1f}%)"
        else:
            row += f" | -"
        print(row)

        # Total time
        values = [stats["total_time"] if stats else None for stats in model_stats]
        row = f"{'Total Time (s)':<25}"
        for val in values:
            row += f" | {val:>15.2f}" if val is not None else f" | {'N/A':>15}"

        if all(v is not None for v in values) and len(values) >= 2:
            diff = values[-1] - values[0]
            pct_change = (diff / values[0] * 100) if values[0] != 0 else 0
            row += f" | {diff:+.2f} ({pct_change:+.1f}%)"
        else:
            row += f" | -"
        print(row)

        print()

    # Summary
    print("="*80)
    print("\n## Summary\n")

    if len(results_list) == 2:
        faster_count = 0
        slower_count = 0
        unchanged_count = 0

        for model_name in all_models:
            model_stats = []
            for results in results_list:
                model_data = next((m for m in results.get("models", []) if m["model"] == model_name), None)
                if model_data:
                    model_stats.append(model_data["averages"]["response_ts"])
                else:
                    model_stats.append(None)

            if all(v is not None for v in model_stats):
                if model_stats[1] > model_stats[0]:
                    faster_count += 1
                elif model_stats[1] < model_stats[0]:
                    slower_count += 1
                else:
                    unchanged_count += 1

        print(f"Comparing {labels[0]} vs {labels[1]}:")
        print(f"  - Faster: {faster_count} models")
        print(f"  - Slower: {slower_count} models")
        print(f"  - Unchanged: {unchanged_count} models")
    else:
        print(f"Compared {len(results_list)} benchmark runs across {len(all_models)} models")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM benchmark results from multiple runs"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON result files to compare (2 or more)"
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels for each file (default: File 1, File 2, ...)"
    )

    args = parser.parse_args()

    if len(args.files) < 2:
        print("✗ Error: Need at least 2 files to compare")
        print("Usage: python3 compare_results.py file1.json file2.json")
        return 1

    # Load all result files
    results_list = []
    for file_path in args.files:
        results = load_json_results(file_path)
        results_list.append(results)

    # Generate labels
    if args.labels and len(args.labels) == len(args.files):
        labels = args.labels
    else:
        labels = [f"Run {i+1}" for i in range(len(args.files))]
        # Try to use dates from filenames if available
        for idx, file_path in enumerate(args.files):
            filename = Path(file_path).stem
            if "_" in filename:
                # Extract potential timestamp
                parts = filename.split("_")
                if len(parts) >= 2:
                    labels[idx] = f"Run {parts[-2]}_{parts[-1]}"

    # Compare results
    compare_results(results_list, labels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
