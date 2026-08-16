"""
Disk cache for semantic-pass model calls.

Why it exists
-------------
The benchmark re-runs the same files against the same prompt many times while
scoring is iterated on. Without a cache every re-run re-bills and re-waits;
with it, one set of API calls can be post-processed any number of ways -- which
is exactly what the cap-on/cap-off ablation needs: same raw model outputs,
two grounding configurations, measurable delta.

Trust model
-----------
The key is the sha256 of the EXACT (system, user) pair sent to the model.
Consequences, by construction:

  * bumping the prompt (any character of the system prompt) changes the key,
    so a new prompt version can never be served a stale response;
  * changing one character of the audited source changes the key likewise;
  * the cache stores the raw model text verbatim -- grounding still runs on
    every read, so a cached response gets no shortcut through validation.

A corrupt or unreadable cache file is treated as a miss and overwritten;
degraded, never crashed, same policy as the rest of the pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import time


def response_key(system: str, user: str) -> str:
    """sha256 over the exact request pair; the null byte keeps the two parts
    from concatenating ambiguously."""
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()


class CachingClient:
    """CompletionClient wrapper that memoizes complete() on disk.

    Wraps any object with a `complete(system, user) -> str` method. Counters
    `hits` and `misses` are public so callers can report cache behaviour the
    same way rejection and cap rates are reported: measured, not assumed.
    """

    def __init__(self, inner, cache_dir: str) -> None:
        self._inner = inner
        self._dir = cache_dir
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.json")

    def complete(self, system: str, user: str) -> str:
        key = response_key(system, user)
        path = self._path(key)

        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    entry = json.load(handle)
                raw_text = entry["raw_text"]
            except (OSError, ValueError, KeyError):
                raw_text = None  # corrupt entry: fall through to a miss
            if raw_text is not None:
                self.hits += 1
                return raw_text

        raw_text = self._inner.complete(system, user)
        self.misses += 1
        self._store(key, path, raw_text)
        return raw_text

    def _store(self, key: str, path: str, raw_text: str) -> None:
        """Write-through, atomically enough for one local user (tmp + replace)."""
        os.makedirs(self._dir, exist_ok=True)
        entry = {
            "key": key,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw_text": raw_text,
        }
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, indent=2)
        os.replace(tmp_path, path)

    def stats(self) -> str:
        return f"cache: {self.hits} hit(s), {self.misses} miss(es) -> {self._dir}"


__all__ = ["CachingClient", "response_key"]
