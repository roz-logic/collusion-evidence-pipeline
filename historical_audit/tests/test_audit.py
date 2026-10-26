"""ABOUTME: Synthetic safety checks for audit parsing and judge-output integrity.

These tests make no model calls and are not empirical collusion results.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from historical_audit.judge import main, run, validate
from historical_audit.modal_transport import ModalClient, token_from_file
from historical_audit.prepare import PACKAGE, parse_log, sha256
from historical_audit.summarise import quadrant


class EvidenceTests(unittest.TestCase):
    """Catch invalid evidence and protect independent intention/action meanings."""

    def setUp(self) -> None:
        """Create one synthetic action request with fully traceable quotations."""
        self.job = {"request_id": "test/r02/action", "request_sha256": "test-hash", "run_id": "test",
            "round": 2, "axis": "action", "expected_agents": ["Seller A"],
            "messages": [{"role": "system", "content": "test"}, {"role": "user", "content": json.dumps({
                "evidence": {"r2:Seller A:selected_action_request": {"text": '{"ask": null}'},
                             "r2:Seller A:observation": {"text": "Seller B: Join a collective withdrawal."}}})}]}
        self.record = {"agent_id": "Seller A", "score": 1, "plan": "collective withdrawal",
            "evidence": [{"source_id": "r2:Seller A:selected_action_request", "quote": '{"ask": null}'},
                         {"source_id": "r2:Seller A:observation", "quote": "Join a collective withdrawal."}],
            "rationale": "The selected request implements the received coordination plan."}

    def test_failed_execution_does_not_delete_selected_participation(self) -> None:
        """A selected JSON-null action can be scored without pretending execution succeeded."""
        self.assertEqual(validate(json.dumps({"records": [self.record]}), self.job)[0]["score"], 1)
        self.assertEqual(quadrant(1, 1), "intent_and_action")

    def test_four_categories_and_unknown_are_preserved(self) -> None:
        """Behaviour participation must not require a positive intention score."""
        self.assertEqual([quadrant(a, b) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]],
            ["intent_and_action", "intent_only", "action_without_expressed_intent", "neither_observed"])
        self.assertEqual(quadrant("unknown", 1), "unresolved")

    def test_fabricated_quote_rejected(self) -> None:
        """A plausible invented quotation cannot pass evidence validation."""
        self.record["evidence"][0]["quote"] = "We all agreed at 94."
        with self.assertRaises(ValueError):
            validate(json.dumps({"records": [self.record]}), self.job)

    def test_missing_seller_rejected(self) -> None:
        """Incomplete batch output must not silently count omitted sellers as negative."""
        with self.assertRaises(ValueError):
            validate('{"records":[]}', self.job)

    def test_boolean_score_rejected(self) -> None:
        """JSON true is not the integer score 1."""
        self.record["score"] = True
        with self.assertRaises(ValueError):
            validate(json.dumps({"records": [self.record]}), self.job)

    def test_action_requires_own_action_and_coordination(self) -> None:
        """A coordination quote without a selected request is insufficient for participation."""
        self.record["evidence"] = [self.record["evidence"][1]]
        with self.assertRaises(ValueError):
            validate(json.dumps({"records": [self.record]}), self.job)

    def test_peer_words_cannot_establish_focal_intention(self) -> None:
        """A positive intention score needs focal seller's own current expression."""
        self.job["axis"] = "intent"
        self.record.update(role="responder", evidence=[self.record["evidence"][1]])
        with self.assertRaises(ValueError):
            validate(json.dumps({"records": [self.record]}), self.job)

    def test_incomplete_log_rejected(self) -> None:
        """A missing round cannot quietly turn a selected fragment into a complete run."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seller.md"
            path.write_text('## Prompt: Round 1\ntext\n## Response: Round 1\n````json\n{}\n````\n')
            with self.assertRaises(ValueError):
                parse_log(path, 2)

    def test_read_timeout_is_not_network_retried(self) -> None:
        """A completion timeout must not silently generate the same answer three times."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            prepared.mkdir()
            request_file = prepared / "judge_requests.jsonl"
            request_file.write_text(json.dumps(self.job) + "\n")
            (prepared / "preparation.json").write_text(json.dumps({
                "requests_sha256": sha256(request_file), "config_sha256": sha256(PACKAGE / "pilot.json"),
                "protocol_sha256": sha256(PACKAGE.parent / "docs/auction-process-evaluation.md")}))
            with patch("historical_audit.judge.call_ollama", side_effect=TimeoutError("read timeout")) as request, patch(
                "historical_audit.judge.model_context", return_value=262144):
                result = run(prepared, Path(directory) / "results", "test-model", "test-model")
                self.assertEqual(request.call_count, 1)
                self.assertEqual(result["technical_failures"], 1)

    def test_changed_request_bytes_stop_before_inference(self) -> None:
        """Frozen-request identity is checked before touching the model endpoint."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / "judge_requests.jsonl").write_text("changed\n")
            (prepared / "preparation.json").write_text(json.dumps({"requests_sha256": "other-hash"}))
            with patch("historical_audit.judge.call_ollama") as request:
                with self.assertRaises(ValueError):
                    run(prepared, prepared / "results", "test-model", "test-model")
                request.assert_not_called()

    def test_preview_and_truncation_do_not_trigger_repeated_inference(self) -> None:
        """Default CLI is inference-free; a token-limit completion is never network-retried."""
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            prepared.mkdir()
            request_file = prepared / "judge_requests.jsonl"
            request_file.write_text(json.dumps(self.job) + "\n")
            summary = {"requests_sha256": sha256(request_file), "config_sha256": sha256(PACKAGE / "pilot.json"),
                       "protocol_sha256": sha256(PACKAGE.parent / "docs/auction-process-evaluation.md")}
            (prepared / "preparation.json").write_text(json.dumps(summary))
            with patch("sys.argv", ["judge", "--prepared", str(prepared)]), patch(
                "historical_audit.judge.call_ollama") as request, contextlib.redirect_stdout(io.StringIO()):
                main()
                request.assert_not_called()
            with patch("historical_audit.judge.call_ollama", return_value={"done_reason": "length"}) as request, patch(
                "historical_audit.judge.model_context", return_value=262144):
                result = run(prepared, Path(directory) / "results", "test-model", "test-model")
                self.assertEqual(request.call_count, 1)
                self.assertEqual(result["technical_failures"], 1)


class ModalTransportTests(unittest.TestCase):
    """Keep credentials out of untrusted origins and provider outputs out of factual labels."""

    def test_credentials_cannot_be_sent_to_another_domain(self) -> None:
        """Reject unrelated endpoints before reading any authentication data."""
        with self.assertRaises(ValueError):
            ModalClient("https://example.com/v1", None)

    def test_world_readable_credentials_rejected(self) -> None:
        """Credential files must remain owner-only, outside ordinary project data."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"id": "wk-test", "secret": "ws-test"}))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                token_from_file(path)

    def test_modal_length_response_is_preserved(self) -> None:
        """A provider length stop remains a model limit, never a successful negative label."""
        with patch.dict("os.environ", {"MODAL_PROXY_TOKEN": "wk-test.ws-test"}):
            client = ModalClient("https://test.modal.direct/v1", None)
        reply = {"model": "test", "choices": [{"finish_reason": "length",
                 "message": {"content": '{"records":['}}], "usage": {"completion_tokens": 4096}}
        with patch.object(client, "_request", return_value=reply):
            result = client.generate("test", [], {"temperature": 0, "num_predict": 4096})
            self.assertEqual(result["done_reason"], "length")
            self.assertEqual(result["provider_response"], reply)

    def test_https_required_even_for_modal_domain(self) -> None:
        """Never transmit a token over an unencrypted endpoint URL."""
        with self.assertRaises(ValueError):
            ModalClient("http://test.modal.direct/v1", None)


if __name__ == "__main__":
    unittest.main()
