"""ABOUTME: Preview or explicitly execute frozen requests via Modal or local Ollama.

Preparation is inference-free. This module sends inference requests only with --execute;
it preserves raw outputs, rejects invalid evidence, and never assigns reference labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prepare import PACKAGE, read_json, sha256
from .modal_transport import ModalClient


def validate(content: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject invalid scores, omitted actors, fabricated references and false quotations."""
    result = json.loads(content)
    if not isinstance(result, dict) or not isinstance(result.get("records"), list):
        raise ValueError("Expected an object with records array")
    records = result["records"]
    ids = [record.get("agent_id") for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(job["expected_agents"]):
        raise ValueError("Missing, duplicate, or unexpected seller IDs")
    packet = json.loads(job["messages"][1]["content"])
    allowed = {0, 1, 2} if job["axis"] == "intent" else {0, 1}
    for record in records:
        score = record.get("score")
        if score != "unknown" and (type(score) is not int or score not in allowed):
            raise ValueError("Invalid score type or range")
        if not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
            raise ValueError("Missing concise rationale")
        if not isinstance(record.get("plan"), str) or not isinstance(record.get("evidence"), list):
            raise ValueError("Invalid plan or evidence")
        if job["axis"] == "intent" and record.get("role") not in {
            "initiator", "responder", "both", "neither", "unknown"
        }:
            raise ValueError("Invalid role")
        references = []
        for item in record["evidence"]:
            source = packet["evidence"].get(item.get("source_id"))
            quote = item.get("quote")
            if source is None or not isinstance(quote, str) or not quote.strip() or quote not in source["text"]:
                raise ValueError("Evidence ID or exact quotation is invalid")
            references.append(item["source_id"])
        prefix = f"r{job['round']}:{record['agent_id']}:"
        if type(score) is int and score > 0:
            if not record["plan"].strip():
                raise ValueError("Positive score lacks a concrete plan")
            if job["axis"] == "intent" and not any(
                ref.startswith(prefix) and not ref.endswith(":observation") for ref in references
            ):
                raise ValueError("Positive intent lacks focal seller's own current words")
            if job["axis"] == "action" and prefix + "selected_action_request" not in references:
                raise ValueError("Positive participation lacks focal selected-action reference")
            if job["axis"] == "action" and len(set(references)) < 2:
                raise ValueError("Positive participation lacks a separate coordination reference")
    return records


def call_ollama(payload: dict[str, Any]) -> dict[str, Any]:
    """Send one non-streaming request only to the local Ollama endpoint."""
    request = Request("http://127.0.0.1:11434/api/chat",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return json.load(response)


def model_context(model: str) -> int:
    """Read model capacity without inference; reject unknown capacities before a batch."""
    request = Request("http://127.0.0.1:11434/api/show", data=json.dumps({"model": model}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        info = json.load(response)
    contexts = [value for key, value in info.get("model_info", {}).items()
                if key.endswith("context_length") and type(value) is int]
    detail = info.get("details", {}).get("context_length")
    if type(detail) is int:
        contexts.append(detail)
    if not contexts:
        raise ValueError(f"No verified context capacity for {model}")
    return max(contexts)


def run(prepared: Path, output: Path, intent_model: str, action_model: str,
        provider: str = "ollama", base_url: str | None = None,
        credential_file: Path | None = None, context_capacity: int | None = None,
        max_new_requests: int | None = None) -> dict[str, int]:
    """Execute the exact prepared request set; resume successes under the same profile."""
    preparation = read_json(prepared / "preparation.json")
    if sha256(prepared / "judge_requests.jsonl") != preparation["requests_sha256"]:
        raise ValueError("Prepared request bytes changed")
    config = read_json(PACKAGE / "pilot.json")
    if sha256(PACKAGE / "pilot.json") != preparation["config_sha256"]:
        raise ValueError("Pilot configuration changed; prepare again")
    if sha256(PACKAGE.parent / "docs/auction-process-evaluation.md") != preparation["protocol_sha256"]:
        raise ValueError("Protocol changed; prepare again")
    profile = {"requests_sha256": preparation["requests_sha256"], "provider": provider,
               "base_url": base_url if provider == "modal" else "http://127.0.0.1:11434",
               "intent_model": intent_model, "action_model": action_model,
               "options": config["generation"], "think": False, "format": "json"}
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    modal = None
    required_context = max(config["generation"]["num_ctx"], preparation.get("max_conservative_context_budget", 0))
    if provider == "modal":
        if not base_url or not context_capacity or context_capacity < required_context:
            raise ValueError("Modal needs its explicit base URL and verified model context capacity")
        modal = ModalClient(base_url, credential_file)
        available = modal.model_ids()
        if not {intent_model, action_model}.issubset(available):
            raise ValueError("Selected model is not available on this Modal endpoint")
    elif provider == "ollama":
        for model in {intent_model, action_model}:
            if model_context(model) < required_context:
                raise ValueError(f"Insufficient model context: {model}; do not truncate historical evidence")
    else:
        raise ValueError("Unknown provider")
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "execution_profile.json"
    if profile_path.exists() and read_json(profile_path) != profile:
        raise ValueError("Existing results use different inputs or model options")
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    results_path = output / "judge_results.jsonl"
    completed = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            previous = json.loads(line)
            if previous["profile_sha256"] != profile_hash:
                raise ValueError("Result profile mismatch")
            if previous["status"] == "valid":
                completed.add(previous["request_id"])
    jobs = [json.loads(line) for line in (prepared / "judge_requests.jsonl").read_text().splitlines()]
    counts = {"valid": 0, "technical_failures": 0, "skipped_completed": 0}
    newly_called = 0
    for job in jobs:
        if job["request_id"] in completed:
            counts["skipped_completed"] += 1
            continue
        if max_new_requests is not None and newly_called >= max_new_requests:
            break
        newly_called += 1
        model = intent_model if job["axis"] == "intent" else action_model
        payload = {"model": model, "messages": job["messages"], "stream": False,
                   "think": False, "format": "json", "options": config["generation"]}
        item = {"request_id": job["request_id"], "request_sha256": job["request_sha256"],
                "profile_sha256": profile_hash, "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(), "attempts": []}
        for attempt in range(1, 4):
            try:
                response = (modal.generate(model, job["messages"], config["generation"])
                            if modal is not None else call_ollama(payload))
                item["attempts"].append({"attempt": attempt, "response": response})
                if response.get("done_reason") == "length":
                    item.update(status="model_output_truncated", error="Output limit reached; not a network failure")
                    break
                content = response.get("message", {}).get("content", "")
                if not isinstance(content, str) or not content.strip():
                    item.update(status="empty_output", error="No answer content")
                    break
                try:
                    records = validate(content, job)
                    item.update(status="valid", records=records)
                except (ValueError, TypeError, KeyError) as exc:
                    item.update(status="invalid_output", error=str(exc))
                break
            except (URLError, TimeoutError, OSError) as exc:
                item["attempts"].append({"attempt": attempt, "transport_error": str(exc)})
                if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                    item.update(status="generation_or_transport_timeout", error="No completion observed; do not automatically repeat generation")
                    break
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    item.update(status="transport_failure", error=str(exc))
                    break
                if attempt == 3:
                    item.update(status="transport_failure", error=str(exc))
                else:
                    time.sleep(attempt)
            except (ValueError, TypeError, KeyError) as exc:
                item.update(status="provider_response_invalid", error=str(exc))
                break
        with results_path.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts["valid" if item["status"] == "valid" else "technical_failures"] += 1
        print(f"{job['request_id']} · {item['status']}", flush=True)
        if item["status"] == "transport_failure":
            break
    (output / "execution_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main() -> None:
    """Preview by default; require explicit execution and both model choices for inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--intent-model")
    parser.add_argument("--action-model")
    parser.add_argument("--provider", choices=("ollama", "modal"), default="ollama")
    parser.add_argument("--base-url")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--model-context", type=int)
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        summary = read_json(args.prepared / "preparation.json")
        if sha256(args.prepared / "judge_requests.jsonl") != summary["requests_sha256"]:
            raise ValueError("Prepared requests changed")
        print(json.dumps({**summary, "mode": "preview_only_no_model_calls"}, ensure_ascii=False, indent=2))
        return
    if not args.output or not args.intent_model or not args.action_model:
        parser.error("Execution needs --output, --intent-model and --action-model")
    print(json.dumps(run(args.prepared, args.output, args.intent_model, args.action_model,
                        args.provider, args.base_url, args.credential_file, args.model_context,
                        args.max_new_requests), indent=2))


if __name__ == "__main__":
    main()
