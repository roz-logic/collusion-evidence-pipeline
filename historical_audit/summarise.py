"""ABOUTME: Describe repeated auction judgments without treating them as reference truth.

Owns exact-input joins, per-seller quote validation and stability denominators.
Keeps missing, invalid and unknown ratings separate; never fits Oz's detector rule.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .codex_batch import validate_records
from .prepare import read_json, section, sha256, write_csv


def quadrant(intent: int | str, action: int | str) -> str:
    """Keep inadequate or missing evidence outside the four fully scored combinations."""
    if intent in ("unknown", "", None) or action in ("unknown", "", None):
        return "unresolved"
    return {(True, True): "intent_and_action", (True, False): "intent_only",
            (False, True): "action_without_expressed_intent",
            (False, False): "neither_observed"}[(int(intent) > 0, int(action) == 1)]


def stability(values: list[int | str | None], expected: int) -> str:
    """Describe agreement only with complete valid repetitions; missing is never a zero."""
    if len(values) != expected or any(v is None for v in values):
        return "incomplete_valid_records"
    if len(set(values)) == 1:
        return "stable_unknown" if values[0] == "unknown" else "stable_known_score"
    return "known_unknown_mixture" if "unknown" in values else "different_known_scores"


def facts(path: Path) -> list[dict[str, Any]]:
    """Read BOM-safe factual CSVs."""
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarise(prepared: Path, judged: Path, output: Path, basic_validation: bool = True) -> dict[str, Any]:
    """Revalidate all receipts and produce full-repetition tables and descriptive summaries."""
    prep = read_json(prepared / "preparation.json")
    profile = read_json(judged / "execution_profile.json")
    if sha256(prepared / "judge_requests.jsonl") != profile["requests_sha256"] or prep["requests_sha256"] != profile["requests_sha256"]:
        raise ValueError("Request identity differs")
    for name, expected in prep["files_sha256"].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f"Facts changed: {name}")
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    jobs = {j["request_id"]: j for j in (json.loads(line) for line in (prepared / "judge_requests.jsonl").read_text().splitlines())}
    repetitions = int(profile.get("repetitions", 1))
    receipts = [json.loads(line) for line in (judged / "judge_results.jsonl").read_text().splitlines()]
    seen, scores, raw_scores, errors = set(), {}, {}, []
    status_counts = Counter()
    receipt_status_counts = Counter()
    quote_failures = Counter()
    for receipt in receipts:
        call_id = receipt.get("call_id", receipt["request_id"])
        if call_id in seen:
            raise ValueError("Repeated call ID in result receipts")
        seen.add(call_id)
        if receipt["profile_sha256"] != profile_hash:
            raise ValueError("Mixed execution profiles")
        job = jobs[receipt["request_id"]]
        if job["request_sha256"] != receipt["request_sha256"]:
            raise ValueError("Receipt belongs to another request")
        repeat = receipt.get("repeat", 1)
        receipt_status_counts[receipt["status"]] += 1
        content = receipt.get("content", "")
        if not content and receipt.get("attempts"):
            content = receipt["attempts"][-1].get("response", {}).get("message", {}).get("content", "")
        if receipt["status"] not in ("valid", "partially_valid"):
            status_counts[receipt["status"]] += 1
            errors.append({"call_id": call_id, "error": receipt.get("error", receipt["status"])})
            continue
        validation_job = {**job, "validation_mode": "basic"} if basic_validation else job
        valid, invalid = validate_records(content, validation_job)
        status_counts["valid" if not invalid else "partially_valid"] += 1
        for record in json.loads(content)["records"]:
            raw_scores[(job["run_id"], job["round"], record["agent_id"], job["axis"], repeat)] = record
        for record in valid:
            scores[(job["run_id"], job["round"], record["agent_id"], job["axis"], repeat)] = record
        for error in invalid:
            quote_failures[error["error"]] += 1
            errors.append({"call_id": call_id, **error})
    source_rows = facts(prepared / "seller_round_facts.csv")
    rows, stable_rows = [], []
    run_counts: dict[str, Counter[str]] = defaultdict(Counter)
    axis_counts: dict[str, Counter[str]] = defaultdict(Counter)
    presence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source in source_rows:
        key = (source["run_id"], int(source["round"]), source["agent_id"])
        values: dict[str, list[Any]] = {"intent": [], "action": []}
        quadrants = []
        for repeat in range(1, repetitions + 1):
            a = scores.get((*key, "intent", repeat), {})
            b = scores.get((*key, "action", repeat), {})
            row = dict(source)
            row.update(repeat=repeat, intent_score=a.get("score", ""), action_participation=b.get("score", ""),
                raw_intent_score=raw_scores.get((*key, "intent", repeat), {}).get("score", ""),
                raw_action_score=raw_scores.get((*key, "action", repeat), {}).get("score", ""),
                intent_rationale=a.get("rationale", ""), action_rationale=b.get("rationale", ""),
                intent_evidence=json.dumps(a.get("evidence", []), ensure_ascii=False),
                action_evidence=json.dumps(b.get("evidence", []), ensure_ascii=False),
                process_category=quadrant(a.get("score"), b.get("score")))
            rows.append(row)
            quadrants.append(row["process_category"])
            run_counts[source["run_id"]][row["process_category"]] += 1
            values["intent"].append(a.get("score"))
            values["action"].append(b.get("score"))
        stable = dict(source)
        for axis in ("intent", "action"):
            label = stability(values[axis], repetitions)
            axis_counts[axis][label] += 1
            presence = [int(v > 0) if type(v) is int else v for v in values[axis]]
            presence_label = stability(presence, repetitions)
            presence_counts[axis][presence_label] += 1
            stable[axis + "_presence_stability"] = presence_label
            distribution = Counter(str(v) if v is not None else "invalid_or_missing" for v in values[axis])
            stable[axis + "_scores"] = json.dumps(values[axis])
            stable[axis + "_distribution"] = json.dumps(distribution, sort_keys=True)
            stable[axis + "_stability"] = label
        stable["quadrant_repeats"] = json.dumps(quadrants)
        stable["joint_stability"] = ("incomplete_valid_records" if "unresolved" in quadrants else
                                      "stable_quadrant" if len(set(quadrants)) == 1 else "different_quadrants")
        stable_rows.append(stable)
    run_rows = facts(prepared / "run_facts.csv")
    for row in run_rows:
        for category in ("intent_and_action", "intent_only", "action_without_expressed_intent", "neither_observed", "unresolved"):
            row[category + "_repeated_ratings"] = run_counts[row["run_id"]][category]
        selected = [r for r in stable_rows if r["run_id"] == row["run_id"]]
        row["unique_seller_round_units"] = len(selected)
        for axis in ("intent", "action"):
            for label in ("stable_known_score", "stable_unknown", "known_unknown_mixture", "different_known_scores", "incomplete_valid_records"):
                row[axis + "_" + label] = sum(r[axis + "_stability"] == label for r in selected)
        deliveries = 0
        delivery_source = ""
        for job in jobs.values():
            if job["run_id"] != row["run_id"] or job["axis"] != "intent":
                continue
            packet = json.loads(job["messages"][1]["content"])
            if not packet["prior_market_rounds"]:
                continue
            previous_messages = packet["prior_market_rounds"][-1].get("seller_messages", {})
            for actor in job["expected_agents"]:
                source = packet["evidence"][f"r{job['round']}:{actor}:observation"]
                received = section(source["text"], "# Messages Received from Other Sellers") or ""
                for sender, message in previous_messages.items():
                    if sender != actor and message:
                        if f"- From {sender}: {message}" not in received:
                            raise ValueError("Prepared peer-delivery evidence changed")
                        deliveries += 1
                        delivery_source = delivery_source or source["source_url"]
        row["channel_delivered"] = deliveries > 0
        row["verified_peer_message_occurrences"] = deliveries
        row["channel_evidence_source"] = delivery_source
        row["actions_agent_chosen"] = all(r["requested_ask_present"] == "True" for r in selected)
        row["action_evidence_source"] = selected[0]["response_source"]
        row["selected_requests_not_executed_order_observed"] = sum(
            r["execution_status"] == "withdrawal_not_executed_order_observed" for r in selected)
        repeated = [r for r in rows if r["run_id"] == row["run_id"]]
        for axis, field in (("intent", "intent_score"), ("action", "action_participation")):
            known = [r[field] for r in repeated if type(r[field]) is int]
            row[axis + "_known_ratings"] = len(known)
            row[axis + "_positive_ratings"] = sum(v > 0 for v in known)
            row[axis + "_unknown_ratings"] = sum(r[field] == "unknown" for r in repeated)
            row[axis + "_invalid_or_missing_ratings"] = sum(r[field] == "" for r in repeated)
        row["independent_reference_available"] = False
    summary = {"profile": profile, "planned_calls": len(jobs)*repetitions, "recorded_calls": len(receipts),
        "call_statuses": dict(status_counts), "original_receipt_statuses": dict(receipt_status_counts),
        "validation_mode": "basic_reanalysis" if basic_validation else profile.get("validation_mode", "exact_quotes"), "unique_seller_round_units": len(source_rows),
        "planned_axis_ratings": len(source_rows)*2*repetitions, "valid_axis_ratings": len(scores),
        "invalid_record_reasons": dict(quote_failures), "axis_stability": {a: dict(c) for a, c in axis_counts.items()},
        "presence_stability": {a: dict(c) for a, c in presence_counts.items()},
        "run_summaries": run_rows, "independent_reference_labels": 0,
        "interpretation": "Repeated ratings are not independent market trials; stability is not accuracy; conditions are descriptive, not causal."}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seller_round_assessments.csv", rows)
    write_csv(output / "seller_round_stability.csv", stable_rows)
    write_csv(output / "run_assessments.csv", run_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (output / "validation_issues.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2)+"\n")
    return summary


def main() -> None:
    """Analyze saved answers only, without model calls."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--judged",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--strict-quotes",action="store_true",help="Reproduce the retired exact-quotation inclusion rule")
    args=parser.parse_args()
    print(json.dumps(summarise(args.prepared,args.judged,args.output,not args.strict_quotes),ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
