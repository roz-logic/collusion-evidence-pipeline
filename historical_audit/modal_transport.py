"""ABOUTME: Authenticated Modal OpenAI-compatible inference without persistent secrets.

Only HTTPS Modal endpoint origins are accepted; redirects are refused. Credentials
come from an environment variable or an owner-only external file, never result logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class RefuseRedirect(HTTPRedirectHandler):
    """Never forward an authentication header to another destination."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        """Refuse redirects instead of risking credential forwarding."""
        return None


def token_from_file(path: Path | None) -> str:
    """Read credentials without exposing their value in configuration or output."""
    if path is None:
        token = os.environ.get("MODAL_PROXY_TOKEN", "")
    else:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Credential file must be an owner-only regular file")
        credentials = json.loads(path.read_text())
        token = credentials["id"] + "." + credentials["secret"]
    if not token.startswith("wk-") or ".ws-" not in token:
        raise ValueError("Missing Modal proxy token; do not substitute a deployment API token")
    return token


class ModalClient:
    """Use a validated Modal endpoint with a secret kept only in process memory."""

    def __init__(self, base_url: str, credential_file: Path | None) -> None:
        """Bind an explicitly chosen Modal HTTPS endpoint without sending requests."""
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment or not (
            host.endswith(".modal.direct") or host.endswith(".modal.run")
        ):
            raise ValueError("Use the HTTPS /v1 base URL of the authorised Modal endpoint")
        self.base_url = base_url.rstrip("/")
        self._token = token_from_file(credential_file)
        self._opener = build_opener(RefuseRedirect())

    def _request(self, suffix: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send credentials only to the validated origin, without redirects."""
        request = Request(self.base_url + suffix,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
        with self._opener.open(request, timeout=240) as response:
            return json.load(response)

    def model_ids(self) -> list[str]:
        """List accessible models without generating tokens."""
        return [item["id"] for item in self._request("/models").get("data", [])]

    def generate(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate bounded JSON while preserving the provider's original response."""
        response = self._request("/chat/completions", {
            "model": model, "messages": messages, "stream": False,
            "temperature": options["temperature"], "max_tokens": options["num_predict"],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}})
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Modal response contains no completion choice")
        choice = choices[0]
        return {"model": response.get("model", model),
                "done_reason": "length" if choice.get("finish_reason") == "length" else choice.get("finish_reason"),
                "message": {"content": choice.get("message", {}).get("content") or ""},
                "provider_response": response}
