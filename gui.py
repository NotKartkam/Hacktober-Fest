"""
Robust Micro-Agent Harness (RMAH) - Interactive Cyberpunk Web GUI Backend
=========================================================================
Runs a local Flask web server delivering the cyberpunk interactive dashboard.
Supports dynamic user-provided datasets, tables, documents, live tool execution,
and multi-dimensional benchmark comparison matrix visualization.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template_string

from rmah.core.agent import RobustMicroAgent
from rmah.core.constrained_decoder import ConstrainedDecoder
from rmah.core.pre_execution_validator import PreExecutionValidator
from rmah.core.loop_detector import LoopDetector
from rmah.core.context_manager import ContextManager
from rmah.baseline.baseline_agent import BaselineUnmodifiedAgent
from rmah.llm.mock_small_llm import MockSmallLLM
from rmah.tools.dummy_tools import get_default_dummy_tools
from rmah.tools.user_data_store import UserDataStore
from rmah.bench.tasks import get_benchmark_tasks, get_sample_tasks
from rmah.bench.runner import BenchmarkRunner

app = Flask(__name__)

# Active session user data store (starts with rich editable defaults)
ACTIVE_USER_STORE = UserDataStore.create_default()


def read_gui_html() -> str:
    """Load the Cyberpunk dashboard HTML."""
    local_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    artifact_path = os.path.join(
        os.path.expanduser("~"),
        ".gemini", "antigravity", "brain",
        "02bf0a8a-8d65-4cb1-9eb8-f9a0db3bfc8a",
        "rmah_cyberpunk_gui.html"
    )
    if os.path.exists(artifact_path):
        with open(artifact_path, "r", encoding="utf-8") as f:
            return f.read()

    return "<h1>RMAH Cyberpunk Dashboard</h1><p>Template loading...</p>"


@app.route("/")
def index():
    """Serve the Cyberpunk Cockpit GUI."""
    html_content = read_gui_html()
    return render_template_string(html_content)


@app.route("/api/user_data", methods=["GET", "POST"])
def user_data_endpoint():
    """
    GET: Retrieve current active user data (tables, datasets, documents).
    POST: Update user data with custom user-provided tables, datasets, or documents.
    """
    global ACTIVE_USER_STORE
    if request.method == "POST":
        payload = request.json or {}
        if "tables" in payload and isinstance(payload["tables"], dict):
            for t_name, rows in payload["tables"].items():
                ACTIVE_USER_STORE.load_table_from_json(t_name, rows)
        if "datasets" in payload and isinstance(payload["datasets"], dict):
            for d_name, vals in payload["datasets"].items():
                ACTIVE_USER_STORE.load_dataset_from_list(d_name, vals)
        if "documents" in payload and isinstance(payload["documents"], dict):
            for doc_key, text in payload["documents"].items():
                ACTIVE_USER_STORE.load_document(doc_key, text)

        return jsonify({
            "status": "success",
            "message": "User data store updated successfully.",
            "summary": ACTIVE_USER_STORE.get_summary(),
        })

    return jsonify({
        "tables": ACTIVE_USER_STORE.tables,
        "datasets": ACTIVE_USER_STORE.datasets,
        "documents": ACTIVE_USER_STORE.documents,
        "summary": ACTIVE_USER_STORE.get_summary(),
    })


@app.route("/api/tools", methods=["GET"])
def get_tools():
    """Return registered tools bound to the active user data store."""
    tools = get_default_dummy_tools(data_store=ACTIVE_USER_STORE)
    tools_data = []
    for t in tools:
        schema = t.get_json_schema()
        tools_data.append({
            "name": t.name,
            "description": t.description,
            "schema": schema,
        })
    return jsonify({"tools": tools_data})


@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """Return full batch of deterministic benchmark tasks."""
    tasks = get_benchmark_tasks(count=50)
    return jsonify({"tasks": [t.model_dump() for t in tasks]})


@app.route("/api/derive_expected_answer", methods=["POST"])
def derive_expected_answer():
    """
    Dynamically look up or evaluate the expected ground truth answer based on
    task instruction content against benchmark tasks, user data stores, or arithmetic expressions.
    """
    data = request.json or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"expected_answer": "", "matched": False})

    p_norm = prompt.lower().strip()

    # 1. Check against the 50 deterministic benchmark tasks
    bench_tasks = get_benchmark_tasks(count=50)
    for t in bench_tasks:
        if t.prompt.strip().lower() == p_norm:
            return jsonify({
                "expected_answer": t.expected_answer,
                "matched": True,
                "source": "benchmark_exact",
                "task_name": t.name,
            })

    for t in bench_tasks:
        if t.prompt.lower().strip() in p_norm or p_norm in t.prompt.lower().strip():
            return jsonify({
                "expected_answer": t.expected_answer,
                "matched": True,
                "source": "benchmark_fuzzy",
                "task_name": t.name,
            })

    # Token overlap match across benchmark tasks (with numeric ID consistency check)
    stop_words = {"what", "is", "the", "of", "a", "an", "to", "in", "for", "use", "find", "tool", "city", "with", "from"}
    prompt_tokens = set(re.findall(r"\w+", p_norm)) - stop_words
    prompt_numbers = set(re.findall(r"\b\d+\b", prompt))
    best_t = None
    best_score = 0.0
    for t in bench_tasks:
        t_numbers = set(re.findall(r"\b\d+\b", t.prompt))
        # If both contain numbers/IDs, ensure there is an intersection
        if prompt_numbers and t_numbers and not (prompt_numbers & t_numbers):
            continue
        t_tokens = set(re.findall(r"\w+", t.prompt.lower())) - stop_words
        if not t_tokens:
            continue
        overlap = len(prompt_tokens & t_tokens) / len(t_tokens)
        if overlap > best_score:
            best_score = overlap
            best_t = t

    if best_t and best_score >= 0.5:
        return jsonify({
            "expected_answer": best_t.expected_answer,
            "matched": True,
            "source": f"benchmark_semantic_{best_t.category}",
            "task_name": best_t.name,
        })

    # 2. Check Math expressions (e.g. 350 * 12 + 45)
    math_match = re.search(r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)", prompt)
    if math_match and any(w in p_norm for w in ["calc", "math", "sum", "result", "solve", "evaluate"]):
        try:
            expr_str = math_match.group(1).strip()
            val = eval(expr_str, {"__builtins__": {}}, {})
            ans_str = str(int(val) if isinstance(val, float) and val.is_integer() else val)
            return jsonify({"expected_answer": ans_str, "matched": True, "source": "calculator_eval"})
        except Exception:
            pass

    # 3. Check User Data Store: Datasets
    store = ACTIVE_USER_STORE
    for ds_name, vals in store.datasets.items():
        if ds_name in p_norm:
            if any(k in p_norm for k in ["avg", "average", "mean"]):
                avg_val = sum(vals) / len(vals) if vals else 0
                ans_str = str(int(avg_val) if avg_val.is_integer() else round(avg_val, 2))
                return jsonify({"expected_answer": ans_str, "matched": True, "source": f"dataset_{ds_name}_avg"})
            elif any(k in p_norm for k in ["sum", "total"]):
                sum_val = sum(vals)
                ans_str = str(int(sum_val) if sum_val.is_integer() else round(sum_val, 2))
                return jsonify({"expected_answer": ans_str, "matched": True, "source": f"dataset_{ds_name}_sum"})
            elif any(k in p_norm for k in ["max", "highest"]):
                return jsonify({"expected_answer": str(max(vals) if vals else 0), "matched": True, "source": f"dataset_{ds_name}_max"})
            elif any(k in p_norm for k in ["min", "lowest"]):
                return jsonify({"expected_answer": str(min(vals) if vals else 0), "matched": True, "source": f"dataset_{ds_name}_min"})
            elif "count" in p_norm:
                return jsonify({"expected_answer": str(len(vals)), "matched": True, "source": f"dataset_{ds_name}_count"})

    # 4. Check User Data Store: Tables
    for tbl_name, rows in store.tables.items():
        if tbl_name in p_norm:
            ids = re.findall(r"\b\d+\b", prompt)
            for raw_id in ids:
                rec_id = int(raw_id)
                if rec_id in rows:
                    row = rows[rec_id]
                    for col_name, col_val in row.items():
                        if col_name.lower() in p_norm:
                            return jsonify({"expected_answer": str(col_val), "matched": True, "source": f"table_{tbl_name}_{col_name}"})
                    for preferred_col in ["name", "item", "company", "tier", "role"]:
                        if preferred_col in row:
                            return jsonify({"expected_answer": str(row[preferred_col]), "matched": True, "source": f"table_{tbl_name}_rec"})
                    return jsonify({"expected_answer": str(list(row.values())[0]), "matched": True, "source": f"table_{tbl_name}_first"})

    # 5. Check User Data Store: Documents
    for topic, text in store.documents.items():
        if topic in p_norm:
            return jsonify({"expected_answer": text, "matched": True, "source": f"document_{topic}"})

    return jsonify({"expected_answer": "", "matched": False})


@app.route("/api/run_task", methods=["POST"])
def run_task():
    """
    Execute a single task through either RMAH or Baseline with custom scaffolding toggles,
    operating directly over user-provided data.
    """
    data = request.json or {}
    task_prompt = data.get("task", "What is the capital of France?")
    expected = data.get("expected_answer", "Paris")
    mode = data.get("mode", "rmah")
    pillars = data.get("pillars", {})

    # Check if custom user data was supplied inline
    store = ACTIVE_USER_STORE
    if "user_data" in data and isinstance(data["user_data"], dict):
        custom_data = data["user_data"]
        store = UserDataStore()
        for t_name, rows in custom_data.get("tables", {}).items():
            store.load_table_from_json(t_name, rows)
        for d_name, vals in custom_data.get("datasets", {}).items():
            store.load_dataset_from_list(d_name, vals)
        for doc_key, text in custom_data.get("documents", {}).items():
            store.load_document(doc_key, text)

    tools = get_default_dummy_tools(data_store=store)

    # Create model simulator
    llm = MockSmallLLM(
        model_name="SmallLLM-7B-Simulated",
        syntax_error_rate=0.25,
        type_error_rate=0.40,
        loop_tendency_rate=0.35,
        seed=int(time.time()) % 1000,
    )

    if mode == "baseline":
        agent = BaselineUnmodifiedAgent(llm=llm, tools=tools, max_steps=10)
    else:
        decoder = ConstrainedDecoder(strict_syntax_enforcement=pillars.get("constrained", True))
        validator = PreExecutionValidator(max_repair_retries=3) if pillars.get("validator", True) else None
        loop_det = LoopDetector() if pillars.get("loop_detector", True) else None
        context_mgr = ContextManager() if pillars.get("context_manager", True) else None

        agent = RobustMicroAgent(
            llm=llm,
            tools=tools,
            max_steps=10,
            constrained_decoder=decoder,
            pre_execution_validator=validator,
            loop_detector=loop_det,
            context_manager=context_mgr,
        )

    res = agent.run(task=task_prompt, task_id="interactive_task", expected_answer=expected)

    steps_history = []
    for step in res.history:
        steps_history.append({
            "step_number": step.step_number,
            "thought": step.action.thought,
            "tool": step.action.tool,
            "tool_input": step.action.tool_input,
            "observation": step.observation.output,
            "is_error": step.observation.is_error,
            "was_repaired": step.was_repaired,
            "was_loop_detected": step.was_loop_detected,
        })

    return jsonify({
        "status": res.status.value,
        "final_answer": res.final_answer,
        "expected_answer": res.expected_answer,
        "is_correct": res.is_correct,
        "steps_taken": res.steps_taken,
        "total_tool_calls": res.total_tool_calls,
        "syntax_errors": res.syntax_errors,
        "validation_failures": res.validation_failures,
        "auto_repair_attempts": res.auto_repair_attempts,
        "successful_auto_repairs": res.successful_auto_repairs,
        "loop_events": res.loop_events,
        "history": steps_history,
        "working_memory_facts": agent.context_manager.extracted_facts if agent.context_manager else [],
    })


@app.route("/api/run_benchmark", methods=["POST"])
def run_benchmark():
    """
    Run the Micro-Agent-Bench evaluation on sample (5) or full (50) tasks
    and return detailed multi-dimensional comparison matrices.
    """
    data = request.json or {}
    task_count = int(data.get("count", 5))

    tasks = get_sample_tasks(count=5) if task_count <= 5 else get_benchmark_tasks(count=task_count)
    runner = BenchmarkRunner(verbose=False)

    def make_baseline():
        llm = MockSmallLLM(seed=42)
        tools = get_default_dummy_tools(data_store=ACTIVE_USER_STORE)
        return BaselineUnmodifiedAgent(llm=llm, tools=tools, max_steps=10)

    def make_rmah():
        llm = MockSmallLLM(seed=42)
        tools = get_default_dummy_tools(data_store=ACTIVE_USER_STORE)
        return RobustMicroAgent(llm=llm, tools=tools, max_steps=10)

    b_report = runner.evaluate_agent(make_baseline, tasks, harness_name="Baseline Unmodified")
    r_report = runner.evaluate_agent(make_rmah, tasks, harness_name="RMAH Modified")

    matrix = runner.build_comparison_matrix(b_report, r_report, tasks)

    return jsonify({
        "baseline": b_report.metrics.model_dump(),
        "rmah": r_report.metrics.model_dump(),
        "comparison_matrix": matrix.model_dump(),
    })


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    port = 5050
    print("\n" + "=" * 70)
    print(" >>> RMAH // CYBERPUNK INTERACTIVE COCKPIT LAUNCHING")
    print(f" Web Dashboard URL: http://localhost:{port}")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
