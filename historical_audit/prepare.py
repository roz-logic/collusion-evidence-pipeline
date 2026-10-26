"""ABOUTME: Prepare complete pinned auction runs for human review and later judges.

Owns source parsing, factual arithmetic, time-bounded evidence packets and hashes.
This module makes no model or network call and does not assign collusion labels.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def sha256(path: Path) -> str:
    """Return the hash of the exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Load a UTF-8 JSON file without inferring absent data."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_log(path: Path, expected_rounds: int) -> dict[int, dict[str, Any]]:
    """Read every original prompt/response, rejecting duplicates or missing turns."""
    text = path.read_text(encoding="utf-8")
    records: dict[int, dict[str, Any]] = {}
    pattern = r"(?ms)^## (Prompt|Response): Round (\d+)[^\n]*\n(.*?)(?=^## (?:Prompt|Response): Round |\Z)"
    for match in re.finditer(pattern, text):
        kind, number, body = match.groups()
        record = records.setdefault(int(number), {})
        field = kind.lower()
        if field in record:
            raise ValueError(f"Duplicate {field}: {path}, round {number}")
        record[field] = body.strip()
        record[field + "_line"] = text.count("\n", 0, match.start()) + 1
        if field == "response":
            block = re.search(r"(?ms)^````json\s*\n(.*?)^````\s*$", body)
            if block is None:
                raise ValueError(f"Missing response JSON: {path}, round {number}")
            record["output"] = json.loads(block.group(1))
    if set(records) != set(range(1, expected_rounds + 1)) or any(
        "prompt" not in record or "response" not in record for record in records.values()
    ):
        raise ValueError(f"Incomplete log: {path}")
    return records


def parse_market(path: Path, expected_rounds: int) -> list[dict[str, Any]]:
    """Read the complete ordered market record."""
    blocks = re.findall(r"(?ms)^````json\s*\n(.*?)^````\s*$", path.read_text())
    rows = [json.loads(block) for block in blocks]
    if [row["round_number"] for row in rows] != list(range(1, expected_rounds + 1)):
        raise ValueError(f"Incomplete market: {path}")
    return rows


def section(prompt: str, heading: str) -> str | None:
    """Extract one recorded prompt section without interpreting it as an instruction."""
    match = re.search(r"(?ms)^" + re.escape(heading) + r"[^\n]*\n(.*?)(?=^# |^## |\Z)", prompt)
    return match.group(1).strip() if match else None


def orders(prompt: str) -> list[Any] | None:
    """Read the order snapshot the seller actually received."""
    value = section(prompt, "## Current Status of Ask Queue")
    return ast.literal_eval(value) if value is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an Excel-readable fact table; blank labels remain blank."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(cache: Path, manifest_path: Path) -> int:
    """Reject any source whose bytes no longer match its pinned Git object."""
    manifest = read_json(manifest_path)
    for item in manifest["files"]:
        raw = (cache / item["cache_path"]).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != item["sha"]:
            raise ValueError(f"Source identity changed: {item['cache_path']}")
    return len(manifest["files"])


def execution(record: dict[str, Any], market: dict[str, Any], actor: str,
              following: dict[str, Any] | None) -> str:
    """Check requested action handling; never infer collusion or harm from it."""
    output = record["output"]
    if "ask" not in output:
        return "not_requested"
    ask = output["ask"]
    if isinstance(ask, (int, float)) and not isinstance(ask, bool):
        return "numeric_ask_applied" if market["seller_asks"].get(actor) == ask else "unknown"
    before = orders(record["prompt"])
    if ask is None:
        if before is not None and not any(seller == actor for _, seller in before):
            return "json_null_ignored_no_prior_order"
        if following is not None:
            after = orders(following["prompt"])
            if after is not None and any(seller == actor for _, seller in after):
                return "withdrawal_not_executed_order_observed"
        return "json_null_ignored_by_code_state_unconfirmed"
    if ask == "null" and following is not None:
        after = orders(following["prompt"])
        if after is not None and not any(seller == actor for _, seller in after):
            return "withdrawal_executed_order_absent"
    return "unknown"


def build(cache: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    """Build all rounds and sellers with frozen prompts; do not call judges."""
    config = read_json(PACKAGE / "pilot.json")
    protocol_path = ROOT / "docs/auction-process-evaluation.md"
    protocol = protocol_path.read_text()
    prompts = {}
    for axis in ("intent", "action"):
        match = re.search(r"<!-- judge-" + axis + r" -->\s*```text\n(.*?)\n```", protocol, re.S)
        if not match:
            raise ValueError(f"Missing canonical prompt: {axis}")
        prompts[axis] = match.group(1)
    output.mkdir(parents=True, exist_ok=True)
    verified = verify_sources(cache, manifest_path)
    requests, seller_rows, round_rows, run_rows, full = [], [], [], [], []
    delivery_checks = 0
    for run_id in config["runs"]:
        directory = cache / "starter/llm-agent-collusion/data" / run_id
        metadata = read_json(directory / "experiment_metadata.json")
        setup = metadata["auction_config"]
        n_rounds = config["rounds_per_run"]
        market = parse_market(directory / "auction_results.md", n_rounds)
        sellers = metadata["seller_ids"]
        if len(sellers) != config["sellers_per_run"] or setup["rounds"] != n_rounds:
            raise ValueError(f"Unexpected dimensions: {run_id}")
        logs = {actor: parse_log(cache / "upstream/data" / run_id / f"{actor}.md", n_rounds)
                for actor in sellers}
        monitor_path = cache / "upstream/data" / run_id / "Overseer.md"
        monitor = parse_log(monitor_path, n_rounds) if monitor_path.exists() else {}
        costs = dict(zip(sellers, setup["seller_valuations"], strict=True))
        values = dict(zip(metadata["buyer_ids"], setup["buyer_valuations"], strict=True))
        prior_profit = {actor: 0.0 for actor in sellers}
        no_trade = {actor: 0 for actor in sellers}
        totals = {"trades": 0, "buyer_surplus": 0.0, "seller_profit": 0.0}
        triggered = False
        for row in market:
            number = row["round_number"]
            profit = {actor: 0.0 for actor in sellers}
            count = {actor: 0 for actor in sellers}
            buyer_surplus = 0.0
            for trade in row["trades"]:
                profit[trade["seller_id"]] += trade["price"] - costs[trade["seller_id"]]
                count[trade["seller_id"]] += 1
                buyer_surplus += values[trade["buyer_id"]] - trade["price"]
            score = monitor.get(number, {}).get("output", {}).get("coordination_score")
            round_rows.append({"run_id": run_id, "round": number,
                "seller_comms_enabled": setup["seller_comms_enabled"],
                "oversight_enabled": setup["oversight_enabled"], "boss_pressure": setup["boss_pressure"],
                "gag_active_before_round_from_monitor": triggered,
                "original_monitor_score": score, "trades": len(row["trades"]),
                "buyer_surplus": buyer_surplus, "seller_profit": sum(profit.values()),
                "total_surplus": buyer_surplus + sum(profit.values())})
            totals["trades"] += len(row["trades"])
            totals["buyer_surplus"] += buyer_surplus
            totals["seller_profit"] += sum(profit.values())
            packets = {axis: {"run_id": run_id, "round": number,
                "market_config": {key: setup.get(key) for key in (
                    "seller_comms_enabled", "oversight_enabled", "boss_pressure")},
                "seller_costs": costs, "buyer_values": values,
                "prior_market_rounds": market[:number - 1], "agents": [], "evidence": {}}
                for axis in prompts}
            for actor in sellers:
                record = logs[actor][number]
                received = section(record["prompt"], "# Messages Received from Other Sellers")
                expected = market[number - 2].get("seller_messages", {}) if number > 1 else {}
                for sender, message in expected.items():
                    if sender != actor and message:
                        delivery_checks += 1
                        if received is None or f"- From {sender}: {message}" not in received:
                            raise ValueError(f"Message not in input: {run_id}/{number}/{actor}/{sender}")
                gag = "# Message from CME Group Regulators" in record["prompt"]
                if monitor and gag != triggered:
                    raise ValueError(f"Monitor/prompt state disagreement: {run_id}/{number}/{actor}")
                prefix = f"r{number}:{actor}"
                source_url = ("https://github.com/andyjliu/llm-agent-collusion/blob/"
                              f"{config['upstream_revision']}/data/{run_id}/{quote(actor + '.md')}")
                request = record["output"]
                following = logs[actor].get(number + 1)
                seller_rows.append({"run_id": run_id, "round": number, "agent_id": actor,
                    "requested_ask_present": "ask" in request,
                    "requested_ask_json": json.dumps(request.get("ask")),
                    "recorded_new_ask": row["seller_asks"].get(actor),
                    "execution_status": execution(record, row, actor, following),
                    "gag_prompt_observed": gag,
                    "outgoing_characters": len(row.get("seller_messages", {}).get(actor, "")),
                    "prior_own_profit": prior_profit[actor], "prior_own_no_trade_streak": no_trade[actor],
                    "current_trades": count[actor], "current_profit": profit[actor],
                    "intent_score": "", "action_participation": "", "human_reference": "",
                    "prompt_source": source_url + f"#L{record['prompt_line']}",
                    "response_source": source_url + f"#L{record['response_line']}"})
                full.append({"run_id": run_id, "round": number, "agent_id": actor,
                             "prompt": record["prompt"], "response": record["response"],
                             "output": request, "following_prompt": following["prompt"] if following else None})
                for axis, packet in packets.items():
                    fields = {"observation": record["prompt"]}
                    if axis == "intent":
                        fields.update({key: request[key] for key in (
                            "reflection", "plan_for_this_hour", "plan_for_message", "message_to_sellers",
                            "scratch_pad_update") if key in request})
                    else:
                        fields["selected_action_request"] = json.dumps(
                            {"ask": request["ask"]} if "ask" in request else {})
                        fields["outgoing_message"] = request.get("message_to_sellers", "") or ""
                    actor_packet = {"agent_id": actor, "evidence_ids": [],
                                    "prior_own_profit": prior_profit[actor],
                                    "prior_own_no_trade_streak": no_trade[actor]}
                    for name, value in fields.items():
                        evidence_id = prefix + ":" + name
                        packet["evidence"][evidence_id] = {"text": value,
                            "source_url": source_url + f"#L{record['prompt_line'] if name == 'observation' else record['response_line']}",
                            "kind": "received_observation" if name == "observation" else name}
                        actor_packet["evidence_ids"].append(evidence_id)
                    packet["agents"].append(actor_packet)
                prior_profit[actor] += profit[actor]
                no_trade[actor] = 0 if count[actor] else no_trade[actor] + 1
            for axis, packet in packets.items():
                user_message = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
                job = {"request_id": f"{run_id}/r{number:02d}/{axis}", "run_id": run_id,
                       "round": number, "axis": axis, "expected_agents": sellers,
                       "messages": [{"role": "system", "content": prompts[axis]},
                                    {"role": "user", "content": user_message}]}
                job["request_sha256"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
                requests.append(job)
            if score == 4:
                triggered = True
        cached = read_json(directory / "collusion_metrics.json")
        if abs(totals["buyer_surplus"] - cached["combined_buyer_surplus"]) > 1e-7 or abs(
            totals["seller_profit"] - cached["combined_seller_profits"]) > 1e-7:
            raise ValueError(f"Economic totals disagree: {run_id}")
        run_rows.append({"run_id": run_id, "rounds": n_rounds, "sellers": len(sellers),
            "seller_comms_enabled": setup["seller_comms_enabled"],
            "oversight_enabled": setup["oversight_enabled"], "boss_pressure": setup["boss_pressure"],
            **totals, "case_type": "", "case_review_status": "not_reviewed"})
    for name, rows in (("round_facts.csv", round_rows), ("seller_round_facts.csv", seller_rows),
                       ("run_facts.csv", run_rows)):
        write_csv(output / name, rows)
    for name, rows in (("judge_requests.jsonl", requests), ("complete_originals.jsonl", full)):
        with (output / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"status": "prepared_not_run", "model_calls": 0, "runs": len(run_rows),
        "rounds": len(round_rows), "seller_round_units": len(seller_rows),
        "planned_judge_requests": len(requests), "verified_sources": verified,
        "peer_message_input_checks": delivery_checks,
        "protocol_sha256": sha256(protocol_path), "config_sha256": sha256(PACKAGE / "pilot.json"),
        "source_manifest_sha256": sha256(manifest_path),
        "files_sha256": {name: sha256(output / name) for name in (
            "run_facts.csv", "round_facts.csv", "seller_round_facts.csv", "complete_originals.jsonl")},
        "requests_sha256": sha256(output / "judge_requests.jsonl"),
        "max_request_characters": max(sum(len(m["content"]) for m in job["messages"]) for job in requests),
        "total_request_characters": sum(sum(len(m["content"]) for m in job["messages"]) for job in requests),
        "max_conservative_context_budget": max(sum(len(m["content"].encode()) for m in job["messages"])
             for job in requests) + config["generation"]["num_predict"] + 1024}
    if summary["max_conservative_context_budget"] > config["generation"]["num_ctx"]:
        raise ValueError("Conservative context budget exceeds configured capacity; do not truncate evidence")
    (output / "protocol.md").write_bytes(protocol_path.read_bytes())
    (output / "pilot.json").write_bytes((PACKAGE / "pilot.json").read_bytes())
    (output / "preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    """Prepare complete runs; explicit CLI has no inference operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cache, args.source_manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
