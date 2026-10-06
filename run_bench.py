#!/usr/bin/env python3
"""
Micro-Agent-Bench Command-Line Runner
=====================================
Automated evaluation test suite comparing Baseline Unmodified Harness
against Robust Micro-Agent Harness (RMAH) on deterministic tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from rmah.bench.runner import BenchmarkRunner
from rmah.bench.tasks import get_benchmark_tasks, get_sample_tasks
from rmah.baseline.baseline_agent import BaselineUnmodifiedAgent
from rmah.core.agent import RobustMicroAgent
from rmah.llm.mock_small_llm import MockSmallLLM
from rmah.llm.local_client import LocalLLMClient
from rmah.tools.dummy_tools import get_default_dummy_tools


def create_llm(backend: str, model_name: str, base_url: str, seed: int = 42):
    """Factory creating LLM instance based on backend choice."""
    if backend == "local":
        print(f"Connecting to Local LLM at {base_url} with model '{model_name}'...")
        return LocalLLMClient(base_url=base_url, model_name=model_name)
    else:
        # Realistic Small LLM (7B) stochastic simulator
        return MockSmallLLM(
            model_name="SmallLLM-7B-Simulator",
            syntax_error_rate=0.25,
            type_error_rate=0.30,
            loop_tendency_rate=0.35,
            seed=seed,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Micro-Agent-Bench: Benchmark evaluation for small LLM agents."
    )
    parser.add_argument(
        "--mode",
        choices=["ab", "rmah", "baseline"],
        default="ab",
        help="Evaluation mode: 'ab' (side-by-side comparison), 'rmah' only, or 'baseline' only.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=50,
        help="Number of tasks to evaluate (5 for sample demo, 50 for full bench).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Quick run of 5 sample representative tasks.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "local"],
        default="mock",
        help="Model backend: 'mock' (7B stochastic simulator) or 'local' (vLLM/Ollama/llama.cpp).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Model name identifier for local backend.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Local LLM API base URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task execution trace.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="micro_agent_bench_results.json",
        help="Path to export benchmark results as JSON.",
    )

    args = parser.parse_args()

    # Load tasks
    if args.sample or args.tasks <= 5:
        tasks = get_sample_tasks(count=5)
        print(f"Loaded {len(tasks)} sample deterministic tasks for demonstration.")
    else:
        tasks = get_benchmark_tasks(count=args.tasks)
        print(f"Loaded {len(tasks)} deterministic tasks for full benchmark evaluation.")

    runner = BenchmarkRunner(verbose=args.verbose)

    # Agent factories
    def make_baseline():
        llm = create_llm(args.backend, args.model, args.base_url, seed=101)
        tools = get_default_dummy_tools()
        return BaselineUnmodifiedAgent(llm=llm, tools=tools, max_steps=10)

    def make_rmah():
        llm = create_llm(args.backend, args.model, args.base_url, seed=101)
        tools = get_default_dummy_tools()
        return RobustMicroAgent(llm=llm, tools=tools, max_steps=10, max_repair_retries_per_step=3)

    export_data = {}

    if args.mode == "ab":
        baseline_report, rmah_report = runner.run_ab_comparison(
            baseline_factory=make_baseline,
            rmah_factory=make_rmah,
            tasks=tasks,
        )
        export_data = {
            "mode": "ab",
            "baseline": baseline_report.model_dump(),
            "rmah": rmah_report.model_dump(),
        }

    elif args.mode == "rmah":
        print(f"\nEvaluating RMAH Modified Harness on {len(tasks)} tasks...")
        rmah_report = runner.evaluate_agent(make_rmah, tasks, harness_name="RMAH Modified")
        m = rmah_report.metrics
        print("\n" + "=" * 60)
        print(f"RMAH RESULTS (Total Tasks: {len(tasks)})")
        print("=" * 60)
        print(f"End-to-End Success Rate: {m.end_to_end_success_rate}% (Target: >80%)")
        print(f"Syntax Error Rate:       {m.syntax_error_rate}% (Target: 0%)")
        print(f"Loop Incidence:          {m.loop_incidence}%")
        print(f"Recovery Rate:           {m.recovery_rate}%")
        print(f"Average Steps per Task:  {m.average_steps_per_task}")
        print("=" * 60)
        export_data = {"mode": "rmah", "rmah": rmah_report.model_dump()}

    elif args.mode == "baseline":
        print(f"\nEvaluating Baseline Unmodified Harness on {len(tasks)} tasks...")
        baseline_report = runner.evaluate_agent(make_baseline, tasks, harness_name="Baseline Unmodified")
        m = baseline_report.metrics
        print("\n" + "=" * 60)
        print(f"BASELINE RESULTS (Total Tasks: {len(tasks)})")
        print("=" * 60)
        print(f"End-to-End Success Rate: {m.end_to_end_success_rate}%")
        print(f"Syntax Error Rate:       {m.syntax_error_rate}%")
        print(f"Loop Incidence:          {m.loop_incidence}%")
        print(f"Recovery Rate:           {m.recovery_rate}%")
        print(f"Average Steps per Task:  {m.average_steps_per_task}")
        print("=" * 60)
        export_data = {"mode": "baseline", "baseline": baseline_report.model_dump()}

    # Export results
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n[Artifact] Detailed benchmark metrics exported to: {args.output_json}")


if __name__ == "__main__":
    main()
