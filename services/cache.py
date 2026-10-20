"""Small JSON file cache with timestamps and expiry.

Cached values are returned with ``cached=True`` and the original ``cached_at``
timestamp so the UI can label them clearly.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import CACHE_DIR


class FileCache:
    def __init__(self, directory: Path = CACHE_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()[:24]
        return self.directory / f"{namespace}-{digest}.json"

    def get(self, namespace: str, key: str, ttl_s: int) -> dict[str, Any] | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - entry.get("cached_at", 0) > ttl_s:
            return None
        value = entry["value"]
        if isinstance(value, dict):
            value = {**value, "cached": True, "cached_at": entry["cached_at_iso"]}
        return value

    def set(self, namespace: str, key: str, value: Any) -> None:
        now = time.time()
        entry = {
            "cached_at": now,
            "cached_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "value": value,
        }
        try:
            self._path(namespace, key).write_text(json.dumps(entry, default=str))
        except OSError:
            pass  # caching is best-effort

    def clear(self) -> None:
        for p in self.directory.glob("*.json"):
            p.unlink(missing_ok=True)


cache = FileCache()
