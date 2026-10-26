"""ABOUTME: Run frozen auction judge packets through authenticated, isolated Codex CLI.

Owns transport, three independent repetitions, raw receipts and exact-quote validation.
It never edits source evidence, fills reference labels, or calls a paid API fallback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from .judge import validate
from .prepare import PACKAGE, read_json, sha256

CLI = "/opt/homebrew/bin/codex"
MODEL = "gpt-6-luna"


def response_schema(axis: str, basic: bool = False) -> dict[str, Any]:
    """Build the strict final-answer contract; evidence text remains independently checked."""
    score = {"anyOf": [{"type": "integer", "enum": [0, 1, 2] if axis == "intent" else [0, 1]},
                       {"type": "string", "enum": ["unknown"]}]}
    fields = {"agent_id": {"type": "string"}, "score": score,
              "plan": {"type": "string"}, "rationale": {"type": "string"},
              "evidence": {"type": "array", "items": {"type": "object", "properties": {
                  "source_id": {"type": "string"}, "quote": {"type": "string"}},
                  "required": ["source_id", "quote"], "additionalProperties": False}}}
    if basic:
        fields = {key: fields[key] for key in ("agent_id", "score", "rationale")}
    if axis == "intent" and not basic:
        fields["role"] = {"type": "string", "enum": ["initiator", "responder", "both", "neither", "unknown"]}
    return {"type": "object", "properties": {"records": {"type": "array", "items": {
        "type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}},
        "required": ["records"], "additionalProperties": False}


def validate_records(content: str, job: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Validate sellers independently so one bad quote cannot erase four valid ratings."""
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("records"), list):
        raise ValueError("Missing records array")
    records = parsed["records"]
    if not all(isinstance(r, dict) for r in records):
        raise ValueError("Each seller record must be an object")
    ids = [r.get("agent_id") for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(job["expected_agents"]):
        raise ValueError("Missing, duplicate, or unexpected seller IDs")
    valid, errors = [], []
    for record in records:
        try:
            if job.get("validation_mode") == "basic":
                allowed = (0, 1, 2) if job["axis"] == "intent" else (0, 1)
                score = record.get("score")
                if score != "unknown" and (type(score) is not int or score not in allowed):
                    raise ValueError("Invalid score type or range")
                if not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
                    raise ValueError("Missing concise rationale")
            else:
                one_job = {**job, "expected_agents": [record["agent_id"]]}
                validate(json.dumps({"records": [record]}), one_job)
            valid.append(record)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"agent_id": record["agent_id"], "error": str(exc)})
    return valid, errors


def execute(job: dict[str, Any], repeat: int, raw_dir: Path, profile_hash: str,
            timeout: int) -> dict[str, Any]:
    """Make one fresh CLI call; retain raw stdout, stderr and final answer on every outcome."""
    call_id = f"{job['request_id']}/repeat-{repeat}"
    folder = raw_dir / job["run_id"] / f"r{job['round']:02d}-{job['axis']}-repeat-{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    answer_path = folder / "answer.json"
    schema_path = folder / "schema.json"
    schema_path.write_text(json.dumps(response_schema(job["axis"])))
    result: dict[str, Any] = {"call_id": call_id, "request_id": job["request_id"],
        "run_id": job["run_id"], "round": job["round"], "axis": job["axis"], "repeat": repeat,
        "request_sha256": job["request_sha256"], "profile_sha256": profile_hash, "model": MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(), "raw_directory": str(folder)}
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="auction-judge-") as scratch:
        command = [CLI, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "-C", scratch, "--model", MODEL,
                   "-c", 'model_reasoning_effort="medium"', "-c", 'web_search="disabled"',
                   "-c", "project_doc_max_bytes=0", "-c", "developer_instructions=" + json.dumps(job["messages"][0]["content"]),
                   "--disable", "shell_tool", "--disable", "multi_agent", "--disable", "plugins",
                   "--disable", "remote_plugin", "--disable", "skill_search", "--json",
                   "--output-schema", str(schema_path), "--output-last-message", str(answer_path), "-"]
        try:
            completed = subprocess.run(command, input=job["messages"][1]["content"], text=True,
                                       capture_output=True, timeout=timeout, check=False)
            stdout, stderr = completed.stdout, completed.stderr
            result["returncode"] = completed.returncode
        except subprocess.TimeoutExpired as exc:
            def text_value(value: str | bytes | None) -> str:
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            stdout, stderr = text_value(exc.stdout), text_value(exc.stderr)
            result.update(status="cli_timeout", error="CLI did not finish; no automatic generation retry")
        (folder / "events.jsonl").write_text(stdout)
        (folder / "stderr.txt").write_text(stderr)
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    tool_items = [e for e in events if e.get("item", {}).get("type") not in (None, "agent_message", "reasoning")]
    result["tool_items"] = tool_items
    result["usage"] = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
    if "status" in result:
        return result
    if result["returncode"]:
        result.update(status="cli_failure", error=(stderr[-1800:] or stdout[-1800:]))
        return result
    if tool_items:
        result.update(status="unexpected_tool_activity", error="Judge used an unexpected non-message item")
        return result
    content = answer_path.read_text() if answer_path.exists() else ""
    result["content"] = content
    if not content.strip():
        result.update(status="empty_output", error="No final response")
        return result
    try:
        records, errors = validate_records(content, job)
        result.update(status="valid" if not errors else "partially_valid", records=records, record_errors=errors)
    except (ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid_output", error=str(exc))
    return result


def run(prepared: Path, output: Path, repeats: int, concurrency: int, limit: int | None,
        timeout: int) -> dict[str, Any]:
    """Execute all planned repetitions, resuming completed receipts under an immutable profile."""
    prep = read_json(prepared / "preparation.json")
    for path, key in [(prepared / "judge_requests.jsonl", "requests_sha256"),
                      ((prepared / "pilot.json") if (prepared / "pilot.json").exists() else PACKAGE / "pilot.json", "config_sha256"),
                      ((prepared / "protocol.md") if (prepared / "protocol.md").exists() else PACKAGE.parent / "docs/auction-process-evaluation.md", "protocol_sha256")]:
        if sha256(path) != prep[key]:
            raise ValueError(f"Frozen input mismatch: {path.name}")
    version = subprocess.run([CLI, "--version"], text=True, capture_output=True, check=True).stdout.strip()
    profile = {"provider": "codex_cli_chatgpt_login", "model": MODEL, "reasoning_effort": "medium",
        "cli_version": version, "requests_sha256": prep["requests_sha256"], "protocol_sha256": prep["protocol_sha256"],
        "config_sha256": prep["config_sha256"], "repetitions": repeats, "fresh_session_per_call": True,
        "rubric_role": "developer_instructions", "evidence_role": "user", "temperature": "not_exposed_by_cli",
        "max_output_tokens": "not_exposed_by_cli", "wall_timeout_seconds": timeout,
        "tools_disabled": ["shell_tool", "multi_agent", "plugins", "remote_plugin", "skill_search", "web_search"],
        "read_only": True, "ignore_user_config": True}
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    profile_file = output / "execution_profile.json"
    if profile_file.exists() and read_json(profile_file) != profile:
        raise ValueError("Existing execution profile differs")
    profile_file.write_text(json.dumps(profile, indent=2) + "\n")
    results_file = output / "judge_results.jsonl"
    seen = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            receipt = json.loads(line)
            if receipt["profile_sha256"] != profile_hash:
                raise ValueError("Receipt profile mismatch")
            seen.add(receipt["call_id"])
    jobs = [json.loads(line) for line in (prepared / "judge_requests.jsonl").read_text().splitlines()]
    work = [(j, r) for r in range(1, repeats + 1) for j in jobs if f"{j['request_id']}/repeat-{r}" not in seen]
    if limit is not None:
        work = work[:limit]
    print(f"Luna judge · {len(jobs)} frozen packets × {repeats} repeats · {len(seen)} saved · {len(work)} pending", flush=True)
    counts: dict[str, Any] = {}
    completed_count = 0
    stopped = False
    iterator = iter(work)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        def submit_next() -> None:
            try:
                job, repeat = next(iterator)
            except StopIteration:
                return
            pending[pool.submit(execute, job, repeat, output / "raw", profile_hash, timeout)] = (job, repeat)
        for _ in range(concurrency):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                receipt = future.result()
                with results_file.open("a") as handle:
                    handle.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                completed_count += 1
                counts[receipt["status"]] = counts.get(receipt["status"], 0) + 1
                print(f"{len(seen)+completed_count}/{len(jobs)*repeats} · {receipt['call_id']} · {receipt['status']} · {receipt['elapsed_seconds']}s", flush=True)
                error = str(receipt.get("error", "")).lower()
                if receipt["status"] == "cli_failure" and any(term in error for term in ("usage limit", "rate limit", "quota", "not supported", "unauthorized", "401", "429")):
                    stopped = True
                if not stopped:
                    submit_next()
    summary = {"planned": len(jobs)*repeats, "previous_receipts": len(seen), "new_receipts": completed_count, "stopped_on_access_or_limit": stopped, "new_statuses": counts}
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Preview unless explicitly authorized with --execute; no external fallback provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"model": MODEL, "repeats": args.repeats, "mode": "preview_no_inference"}))
        return
    print(json.dumps(run(args.prepared, args.output, args.repeats, args.concurrency,
                         args.max_new_requests, args.timeout), indent=2))

if __name__ == "__main__":
    main()
