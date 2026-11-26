"""Run artifacts: everything needed to recompute metrics or audit a verdict later."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import __version__
from .replay import ReplayResult

RUN_FILE = "run.json"
RECORDS_FILE = "records.json.gz"
NOISE_JSON = "noise.json"
NOISE_MARKDOWN = "noise.md"


def redact_url(url: str) -> str:
    """Drop credentials from a URL before it is persisted."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"  # Preserve a parseable IPv6 authority after removing credentials.
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def client_environment() -> dict[str, Any]:
    return {
        "slogate_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "host_id": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_run(
    directory: str | Path,
    result: ReplayResult,
    *,
    metrics: dict[str, Any],
    fidelity: dict[str, Any],
    meta: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    document = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "meta": meta,
        "client": client_environment(),
        "replay": result.to_dict(include_records=False),
        "fidelity": fidelity,
        "metrics": metrics,
    }
    write_json(directory / RUN_FILE, document)
    records = json.dumps([r.to_dict() for r in result.records], allow_nan=False).encode()
    (directory / RECORDS_FILE).write_bytes(gzip.compress(records, mtime=0))
    return directory / RUN_FILE


def read_run(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / RUN_FILE).read_text(encoding="utf-8"))


def read_records(directory: str | Path) -> list[dict[str, Any]]:
    return json.loads(gzip.decompress((Path(directory) / RECORDS_FILE).read_bytes()))
