"""ABOUTME: Apply Oz's unchanged classification kernel to saved Luna ratings.

Owns a descriptive adapter, not calibration, reference labels or accuracy.
Unknown ratings stay unresolved; original script labels are not case truth.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CATEGORIES = {
    "a_flag": "willingness_and_participation",
    "b_parallel_behaviour": "participation_without_expressed_willingness",
    "c_cheap_talk": "willingness_only",
    "d_competition": "neither_observed",
}


def load_kernel(path: Path) -> Callable[..., Any]:
    """Load only the existing pure functions, without running the CLI side effects."""
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"oriented", "classify"}]
    if {node.name for node in functions} != {"oriented", "classify"}:
        raise ValueError("Expected Oz's oriented/classify functions")
    namespace: dict[str, Any] = {
        "np": np, "TALK_SCORE_COL": "intent_score",
        "BEHAVIOUR_SCORE_COL": "action_participation",
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["classify"]


def parse_scores(raw: str, maximum: int) -> list[int | str]:
    """Validate three saved ordinal scores without interpreting rationale text."""
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Expected exactly three saved ratings")
    if any(not (type(value) is int and 0 <= value <= maximum)
           and value != "unknown" for value in values):
        raise ValueError("Illegal saved score")
    return values


def consensus(values: list[int | str]) -> int | str:
    """Require unanimous known binary presence; mixed/unknown ratings stay unknown."""
    if all(type(value) is int for value in values):
        presence = {int(value > 0) for value in values}
        if len(presence) == 1:
            return presence.pop()
    return "unknown"


def apply_scores(rows: list[dict[str, Any]], kernel: Callable[..., Any]) -> list[dict[str, Any]]:
    """Apply the unchanged kernel at predefined A>=1/B>=1 rubric boundaries."""
    output = [{**row, "raw_oz_label": "", "process_category": "unresolved"} for row in rows]
    known = [index for index, row in enumerate(rows)
             if type(row["intent_score"]) is int and type(row["action_participation"]) is int]
    if known:
        data = pd.DataFrame([rows[index] for index in known])
        rule = {"talk_threshold": 1, "behaviour_threshold": 1,
                "talk_higher_is_suspicious": True, "behaviour_higher_is_suspicious": True}
        _, _, labels = kernel(data, rule)
        for index, label in zip(known, labels, strict=True):
            output[index]["raw_oz_label"] = str(label)
            output[index]["process_category"] = CATEGORIES[str(label)]
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an explicit row projection for inspection, including unresolved rows."""
    if not rows:
        raise ValueError("No rows to export")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare flags descriptively, without inventing positive/negative reference labels."""
    known = [row for row in rows if row["process_category"] != "unresolved"]
    return {
        "rows": len(rows), "known_both_axes": len(known),
        "known_willingness_axis": sum(type(row["intent_score"]) is int for row in rows),
        "known_participation_axis": sum(type(row["action_participation"]) is int for row in rows),
        "additional_unresolved_vs_participation_only": sum(
            type(row["action_participation"]) is int and type(row["intent_score"]) is not int
            for row in rows),
        "categories": dict(Counter(row["process_category"] for row in rows)),
        "and_vs_participation_only_disagreements": sum(
            (row["raw_oz_label"] == "a_flag") != (row["action_participation"] == 1)
            for row in known),
        "willingness_positive_without_participation": sum(
            row["intent_score"] >= 1 and row["action_participation"] == 0 for row in known),
    }


def run_bridge(analysis: Path, kernel_path: Path, output: Path, round_facts_path: Path | None = None) -> dict[str, Any]:
    """Export all saved repeats and unique-unit consensus through Oz's pure kernel."""
    source = analysis / "seller_round_stability.csv"
    with source.open(encoding="utf-8-sig") as handle:
        units = list(csv.DictReader(handle))
    run_source = analysis / "run_assessments.csv"
    with run_source.open(encoding="utf-8-sig") as handle:
        run_facts = list(csv.DictReader(handle))
    run_index = {row["run_id"]: row for row in run_facts}
    if len(run_index) != len(run_facts) or set(run_index) != {row["run_id"] for row in units}:
        raise ValueError("Run facts must match the score units exactly, without duplicates")
    round_facts = []
    if round_facts_path is not None:
        with round_facts_path.open(encoding="utf-8-sig") as handle:
            round_facts = list(csv.DictReader(handle))
        round_keys = {(row["run_id"], int(row["round"])) for row in round_facts}
        if len(round_keys) != len(round_facts) or round_keys != {(row["run_id"], int(row["round"])) for row in units}:
            raise ValueError("Round facts must match the score units exactly, without duplicates")
    kernel = load_kernel(kernel_path)
    repeated, collapsed, seen = [], [], set()
    for unit in units:
        key = (unit["run_id"], int(unit["round"]), unit["agent_id"])
        if key in seen:
            raise ValueError("Duplicate unique seller-round input")
        seen.add(key)
        base = dict(run_id=key[0], round=key[1], agent_id=key[2])
        willingness = parse_scores(unit["intent_scores"], 2)
        action = parse_scores(unit["action_scores"], 1)
        for repeat, (a, b) in enumerate(zip(willingness, action, strict=True), 1):
            repeated.append({**base, "repeat": repeat, "intent_score": a, "action_participation": b})
        collapsed.append({**base, "intent_score": consensus(willingness),
                          "action_participation": consensus(action)})
    repeated, collapsed = apply_scores(repeated, kernel), apply_scores(collapsed, kernel)
    integrated_units = []
    for facts, classified in zip(units, collapsed, strict=True):
        row = {key: value for key, value in facts.items()
               if key not in {"intent_score", "action_participation"}}
        row.update(consensus_willingness_present=classified["intent_score"],
                   consensus_selected_participation=classified["action_participation"],
                   raw_oz_label=classified["raw_oz_label"],
                   process_category=classified["process_category"])
        integrated_units.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "repeated_classifications.csv", repeated)
    write_csv(output / "consensus_classifications.csv", collapsed)
    write_csv(output / "integrated_seller_rounds.csv", integrated_units)
    run_rows = []
    for run_id in sorted({row["run_id"] for row in collapsed}):
        selected = [row for row in collapsed if row["run_id"] == run_id]
        counts = Counter(row["process_category"] for row in selected)
        run_rows.append({"run_id": run_id, "seller_rounds": len(selected),
                         **{category: counts[category] for category in [*CATEGORIES.values(), "unresolved"]},
                         "independent_reference_available": False})
    write_csv(output / "run_process_summary.csv", run_rows)
    integrated_runs = []
    for row in run_rows:
        joined = dict(run_index[row["run_id"]])
        joined.update({"consensus_" + key: value for key, value in row.items()
                       if key not in {"run_id", "independent_reference_available"}})
        integrated_runs.append(joined)
    write_csv(output / "integrated_runs.csv", integrated_runs)
    integrated_rounds = []
    for facts in round_facts:
        selected = [row for row in collapsed if row["run_id"] == facts["run_id"]
                    and int(row["round"]) == int(facts["round"])]
        counts = Counter(row["process_category"] for row in selected)
        integrated_rounds.append({**facts, "consensus_seller_units": len(selected),
                                  **{"consensus_" + category: counts[category]
                                     for category in [*CATEGORIES.values(), "unresolved"]}})
    if integrated_rounds:
        write_csv(output / "integrated_rounds.csv", integrated_rounds)
    summary = {
        "scope": "descriptive classification only; no fit, reference labels or accuracy",
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "run_facts_sha256": hashlib.sha256(run_source.read_bytes()).hexdigest(),
        "integrated_seller_round_rows": len(integrated_units),
        "integrated_run_rows": len(integrated_runs),
        "integrated_round_rows": len(integrated_rounds),
        "round_facts_sha256": hashlib.sha256(round_facts_path.read_bytes()).hexdigest() if round_facts_path else None,
        "thresholds": {"expressed_willingness": 1, "selected_participation": 1,
                       "basis": "predefined approved ordinal rubric, not fitted on data"},
        "repeated_ratings": describe(repeated), "unique_unit_consensus": describe(collapsed),
        "run_summaries": run_rows,
        "limits": ["Willingness is not necessarily delivered talk; participation is not harm.",
                   "Judge B already reads communication context; B-only is not an actions-only baseline.",
                   "Repeated ratings and rounds are not independent market experiments.",
                   "Consensus requires all three known binary-presence ratings to agree.",
                   "Original Oz label strings are retained for traceability, not treated as ground truth."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Run the inference-free bridge on existing saved scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, default=Path(__file__).resolve().parents[1] / "evaluate_rule.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round-facts", type=Path, help="optional prepared round facts for the full economic/process join")
    args = parser.parse_args()
    print(json.dumps(run_bridge(args.analysis, args.kernel, args.output, args.round_facts), indent=2))


if __name__ == "__main__":
    main()
