"""Tests for the on-disk cache of semantic-pass model calls.

The guarantee under test: the cache key commits to the EXACT (system, user)
pair, so a prompt bump or a source change can never be served a stale
response, while identical re-runs never touch the network again. No test here
performs any network call.
"""

from __future__ import annotations

import json
import os

from auditor.cache import CachingClient, response_key
from auditor.llm_pass import load_prompt, make_client_from_env


class CountingStub:
    """Inner client that counts calls and answers deterministically."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return f'{{"findings": [], "echo": "{self.calls}"}}'


def test_second_identical_call_is_served_from_disk(tmp_path):
    inner = CountingStub()
    client = CachingClient(inner, str(tmp_path))

    first = client.complete("SYSTEM", "USER")
    second = client.complete("SYSTEM", "USER")

    assert inner.calls == 1                # the network was hit exactly once
    assert first == second                 # byte-identical replay
    assert (client.hits, client.misses) == (1, 1)


def test_prompt_bump_invalidates_the_cache():
    """One changed character in the system prompt must change the key: a new
    prompt version can never inherit an old version's responses."""
    assert response_key("prompt v1.1.0", "same file") != response_key(
        "prompt v1.2.0", "same file"
    )


def test_source_change_invalidates_the_cache(tmp_path):
    inner = CountingStub()
    client = CachingClient(inner, str(tmp_path))
    client.complete("SYSTEM", "file A")
    client.complete("SYSTEM", "file B")
    assert inner.calls == 2
    assert (client.hits, client.misses) == (0, 2)


def test_cache_entry_is_auditable_json_named_by_its_key(tmp_path):
    client = CachingClient(CountingStub(), str(tmp_path))
    client.complete("S", "U")

    key = response_key("S", "U")
    path = tmp_path / f"{key}.json"
    assert path.is_file()
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["key"] == key
    assert entry["raw_text"].startswith('{"findings"')
    assert "created_at" in entry


def test_corrupt_entry_degrades_to_a_miss_and_heals(tmp_path):
    inner = CountingStub()
    client = CachingClient(inner, str(tmp_path))
    client.complete("S", "U")

    key = response_key("S", "U")
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")

    replay = client.complete("S", "U")     # corrupt -> miss -> re-fetch
    assert inner.calls == 2
    assert replay                          # a real answer came back
    healed = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert healed["raw_text"] == replay    # the entry was overwritten cleanly


def test_make_client_from_env_wraps_in_cache_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.delenv("AUDITOR_OFFLINE", raising=False)
    monkeypatch.delenv("AUDITOR_NO_CACHE", raising=False)
    monkeypatch.setenv("AUDITOR_CACHE_DIR", str(tmp_path))

    client = make_client_from_env(load_prompt())
    assert isinstance(client, CachingClient)


def test_auditor_no_cache_returns_the_bare_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.delenv("AUDITOR_OFFLINE", raising=False)
    monkeypatch.setenv("AUDITOR_NO_CACHE", "1")

    client = make_client_from_env(load_prompt())
    assert not isinstance(client, CachingClient)


def test_stats_line_reports_hits_misses_and_directory(tmp_path):
    client = CachingClient(CountingStub(), str(tmp_path))
    client.complete("S", "U")
    client.complete("S", "U")
    line = client.stats()
    assert "1 hit(s)" in line and "1 miss(es)" in line
    assert os.path.basename(str(tmp_path)) in line
