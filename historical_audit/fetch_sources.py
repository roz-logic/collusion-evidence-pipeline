"""ABOUTME: Download only the pinned public evidence listed in source_manifest.json.

GET-only source acquisition; no model calls, market execution or Git mutations.
Every downloaded file must match its original Git blob before it enters the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

from .prepare import PACKAGE, read_json


def fetch(cache: Path, manifest: Path) -> dict[str, int]:
    """Acquire pinned public source bytes, preserving verified cache hits."""
    totals = {"downloaded": 0, "cached": 0, "bytes_verified": 0, "model_calls": 0}
    for item in read_json(manifest)["files"]:
        destination = cache / item["cache_path"]
        if destination.is_file():
            raw = destination.read_bytes()
            origin = "cached"
        else:
            url = (f"https://raw.githubusercontent.com/{item['repository']}/"
                   f"{item['revision']}/{quote(item['path'])}")
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=60) as response:
                        raw = response.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            origin = "downloaded"
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != item["sha"]:
            raise ValueError(f"Source differs from pinned revision: {item['cache_path']}")
        if origin == "downloaded":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        totals[origin] += 1
        totals["bytes_verified"] += len(raw)
        print(f"{origin} · {item['cache_path']}", flush=True)
    return totals


def main() -> None:
    """Fetch source files only; all inference is a separate explicit operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.cache, PACKAGE / "source_manifest.json"), indent=2))


if __name__ == "__main__":
    main()
