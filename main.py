"""
Robust Micro-Agent Harness (RMAH) Main Entrypoint
=================================================
Provides quick access to run single interactive tasks, run 5 sample benchmark tasks,
or execute the full 50-task Micro-Agent-Bench evaluation suite.
"""

from __future__ import annotations

import sys
from rmah.core.agent import RobustMicroAgent
from rmah.llm.mock_small_llm import MockSmallLLM
from rmah.tools.dummy_tools import get_default_dummy_tools


def run_demo():
    print("=" * 70)
    print(" Robust Micro-Agent Harness (RMAH) - Live Scaffolding Demonstration")
    print("=" * 70)

    tools = get_default_dummy_tools()
    # Simulate a small 7B model prone to syntax and type mistakes
    llm = MockSmallLLM(
        model_name="SmallLLM-7B-Simulated",
        syntax_error_rate=0.20,
        type_error_rate=0.40,
        loop_tendency_rate=0.30,
        seed=42,
    )

    agent = RobustMicroAgent(
        llm=llm,
        tools=tools,
        max_steps=8,
        max_repair_retries_per_step=3,
    )

    sample_task = "Query the database for user record_id 101 in table 'users'. Note: record_id must be strictly an integer."
    print(f"\nTask: {sample_task}\n")
    print("Executing through RMAH with Constrained Decoding & Auto-Repair...")

    result = agent.run(task=sample_task, task_id="demo_01", expected_answer="Alice Johnson")

    print("\n--- Execution Summary ---")
    print(f"Status:                 {result.status.value}")
    print(f"Final Answer:           {result.final_answer}")
    print(f"Steps Taken:            {result.steps_taken}")
    print(f"Total Tool Calls:       {result.total_tool_calls}")
    print(f"Syntax Errors:          {result.syntax_errors} (Target: 0)")
    print(f"Validation Failures:    {result.validation_failures}")
    print(f"Auto-Repair Attempts:   {result.auto_repair_attempts}")
    print(f"Successful Repairs:     {result.successful_auto_repairs}")
    print(f"Loop Interventions:     {result.loop_events}")
    print(f"Correctness Verified:   {result.is_correct}")

    print("\nExecution Step Trace:")
    for step in result.history:
        rep_tag = " [Auto-Repaired]" if step.was_repaired else ""
        loop_tag = " [Loop-Detected]" if step.was_loop_detected else ""
        print(f"  Step {step.step_number}{rep_tag}{loop_tag}:")
        print(f"    Thought: {step.action.thought}")
        print(f"    Action:  {step.action.tool}({step.action.tool_input})")
        print(f"    Obs:     {step.observation.output[:90]}...")

    print("\n" + "=" * 70)
    print("To run the full evaluation suite, execute: python run_bench.py --mode ab")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
