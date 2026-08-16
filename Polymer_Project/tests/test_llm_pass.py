"""Tests for the semantic LLM pass.

No test here touches the network. A stub client returns scripted responses,
which lets the grounding machinery be tested against exactly the failure modes
it exists for: invented line numbers, malformed JSON, out-of-vocabulary
severities, and prose where JSON was demanded.

The definition of done for this module is measurable here: across every
scripted response, no finding citing a nonexistent line is ever ACCEPTED.
"""

from __future__ import annotations

import json

import pytest

from auditor.llm_pass import (
    ground_findings,
    load_prompt,
    number_source,
    parse_model_json,
    semantic_audit,
)

SOURCE = """import pandas as pd

events = pd.read_csv("news.csv")
prices = pd.read_csv("px.csv")
merged = pd.merge_asof(events, prices, on="ts", direction="forward")
signal = merged["close"].pct_change()
"""  # 6 lines


class StubClient:
    """Returns queued responses and records what it was asked."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def make_response(*findings: dict) -> str:
    return json.dumps({"findings": list(findings)})


VALID_FINDING = {
    "line_start": 5,
    "line_end": 5,
    "severity": "high",
    "external_dependency": None,  # declared self-contained: eligible for high
    "explanation": "merge_asof direction='forward' pairs each event with a later price.",
    "suggested_fix": "Use direction='backward'.",
}


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------


def test_prompt_file_loads_and_declares_required_fields():
    spec = load_prompt()
    assert spec["temperature"] == 0.0
    assert spec["version"]
    assert "JSON" in spec["system_prompt"]
    assert "merge_asof" in spec["system_prompt"]


def test_prompt_is_at_least_1_2_0_and_declares_the_dependency_field():
    """v1.2.0 moved the severity-entitlement constraint into the harness; the
    prompt must document the field, the enforcement, and show it in a worked
    example, so the model sees the contract it is being held to."""
    spec = load_prompt()
    version = tuple(int(part) for part in str(spec["version"]).split("."))
    assert version >= (1, 2, 0)
    assert "external_dependency" in spec["system_prompt"]
    assert "capped" in spec["system_prompt"]  # the enforcement is announced
    example_outputs = "".join(ex["output"] for ex in spec.get("examples", []))
    assert "external_dependency" in example_outputs


def test_prompt_excludes_the_ast_rules_territory():
    """The model must not re-report what the deterministic rules own."""
    system = load_prompt()["system_prompt"]
    for owned in ("shift", "center=True", "bfill"):
        assert owned in system  # named precisely so they can be forbidden


def test_numbering_matches_the_announced_format():
    numbered = number_source("a = 1\nb = 2")
    assert numbered.splitlines()[0] == "   1 | a = 1"
    assert numbered.splitlines()[1] == "   2 | b = 2"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_plain_json_is_parsed():
    assert parse_model_json('{"findings": []}') == {"findings": []}


def test_markdown_fences_are_tolerated():
    fenced = "```json\n{\"findings\": []}\n```"
    assert parse_model_json(fenced) == {"findings": []}


def test_prose_output_is_rejected_not_repaired():
    with pytest.raises(ValueError):
        parse_model_json("I found no issues in this file.")


# ---------------------------------------------------------------------------
# Grounding -- the core guarantee
# ---------------------------------------------------------------------------


def test_valid_finding_is_accepted_with_snippet_from_the_real_file():
    accepted, rejected = ground_findings(
        {"findings": [VALID_FINDING]}, SOURCE, "demo.py"
    )
    assert len(accepted) == 1 and not rejected
    finding = accepted[0]
    assert finding.detector == "llm"
    assert finding.rule_id == "SEM"
    # The snippet is line 5 of the REAL source, not whatever the model claimed.
    assert "merge_asof" in finding.snippet


def test_finding_on_a_nonexistent_line_is_rejected():
    bad = dict(VALID_FINDING, line_start=999, line_end=999)
    accepted, rejected = ground_findings({"findings": [bad]}, SOURCE, "demo.py")
    assert accepted == []
    assert len(rejected) == 1 and "outside" in rejected[0]["reason"]


def test_line_zero_is_rejected():
    bad = dict(VALID_FINDING, line_start=0)
    accepted, rejected = ground_findings({"findings": [bad]}, SOURCE, "demo.py")
    assert accepted == [] and rejected


def test_inverted_line_range_is_rejected():
    bad = dict(VALID_FINDING, line_start=5, line_end=2)
    accepted, rejected = ground_findings({"findings": [bad]}, SOURCE, "demo.py")
    assert accepted == [] and "invalid" in rejected[0]["reason"]


def test_non_integer_lines_are_rejected():
    bad = dict(VALID_FINDING, line_start="five")
    accepted, rejected = ground_findings({"findings": [bad]}, SOURCE, "demo.py")
    assert accepted == [] and "non-integer" in rejected[0]["reason"]


def test_unknown_severity_is_coerced_to_review():
    odd = dict(VALID_FINDING, severity="catastrophic")
    accepted, _ = ground_findings({"findings": [odd]}, SOURCE, "demo.py")
    assert accepted[0].severity == "review"


def test_empty_explanation_is_rejected():
    bad = dict(VALID_FINDING, explanation="   ")
    accepted, rejected = ground_findings({"findings": [bad]}, SOURCE, "demo.py")
    assert accepted == [] and "explanation" in rejected[0]["reason"]


def test_mixed_batch_keeps_only_the_grounded_findings():
    batch = {
        "findings": [
            VALID_FINDING,
            dict(VALID_FINDING, line_start=42),
            dict(VALID_FINDING, line_start=2, line_end=3, severity="review"),
        ]
    }
    accepted, rejected = ground_findings(batch, SOURCE, "demo.py")
    assert len(accepted) == 2 and len(rejected) == 1


# ---------------------------------------------------------------------------
# Severity entitlement -- the deterministic cap (v1.2.0)
# ---------------------------------------------------------------------------
# The lesson behind these tests: prompt v1.1.0 asked for exactly this behaviour
# and the model violated it on first contact with real code. A prompt rule is
# a request; ground_findings() is a guarantee. high/medium must be EARNED by
# an explicit "external_dependency": null.


def test_declared_external_dependency_caps_high_to_review():
    dependent = dict(
        VALID_FINDING,
        external_dependency="The P&L convention of the engine consuming pos.",
    )
    accepted, rejected = ground_findings({"findings": [dependent]}, SOURCE, "demo.py")
    assert not rejected and len(accepted) == 1
    finding = accepted[0]
    assert finding.severity == "review"
    assert finding.capped_from == "high"
    assert "P&L convention" in finding.external_dependency


def test_missing_dependency_field_caps_too_omission_buys_nothing():
    legacy = {k: v for k, v in VALID_FINDING.items() if k != "external_dependency"}
    accepted, _ = ground_findings({"findings": [legacy]}, SOURCE, "demo.py")
    assert accepted[0].severity == "review"
    assert accepted[0].capped_from == "high"
    assert accepted[0].external_dependency is None  # nothing was named


def test_explicit_null_dependency_preserves_high():
    accepted, _ = ground_findings({"findings": [VALID_FINDING]}, SOURCE, "demo.py")
    assert accepted[0].severity == "high"
    assert accepted[0].capped_from is None


@pytest.mark.parametrize("token", ["", "   ", "None", "null", "N/A", "-"])
def test_null_like_strings_count_as_no_dependency(token):
    finding = dict(VALID_FINDING, external_dependency=token)
    accepted, _ = ground_findings({"findings": [finding]}, SOURCE, "demo.py")
    assert accepted[0].severity == "high"
    assert accepted[0].capped_from is None


def test_wrong_typed_dependency_is_treated_as_undeclared():
    finding = dict(VALID_FINDING, external_dependency=["a", "list"])
    accepted, _ = ground_findings({"findings": [finding]}, SOURCE, "demo.py")
    assert accepted[0].severity == "review"
    assert accepted[0].capped_from == "high"


def test_review_with_a_dependency_is_not_counted_as_capped():
    """Capping records a DISAGREEMENT with the model; a model that already
    said review agreed with the policy and must not inflate the counter."""
    finding = dict(
        VALID_FINDING, severity="review",
        external_dependency="Whether the loader forward-fills before this runs.",
    )
    accepted, _ = ground_findings({"findings": [finding]}, SOURCE, "demo.py")
    assert accepted[0].severity == "review"
    assert accepted[0].capped_from is None
    assert accepted[0].external_dependency is not None


def test_vocabulary_coercion_and_cap_stay_distinct():
    """An out-of-vocabulary severity is a vocabulary problem, not an
    entitlement problem: coerced to review, but not counted as capped."""
    odd = dict(VALID_FINDING, severity="catastrophic")
    accepted, _ = ground_findings({"findings": [odd]}, SOURCE, "demo.py")
    assert accepted[0].severity == "review"
    assert accepted[0].capped_from is None


def test_cap_can_be_disabled_for_the_benchmark_ablation_only():
    dependent = dict(
        VALID_FINDING, external_dependency="A caller's fill convention."
    )
    accepted, _ = ground_findings(
        {"findings": [dependent]}, SOURCE, "demo.py", enforce_external_cap=False
    )
    assert accepted[0].severity == "high"  # v1.1.0 behaviour, reproducible
    assert accepted[0].capped_from is None


def test_capped_counter_flows_into_the_result_and_its_json():
    dependent = dict(
        VALID_FINDING,
        severity="medium",
        external_dependency="The engine's position/return alignment.",
    )
    client = StubClient(make_response(dependent))
    result = semantic_audit(SOURCE, "demo.py", client=client)
    assert len(result.capped) == 1
    entry = result.capped[0]
    assert entry["capped_from"] == "medium"
    assert "alignment" in entry["external_dependency"]
    assert result.to_dict()["capped"] == result.capped
    assert any("capped" in note for note in result.notes)


# ---------------------------------------------------------------------------
# End-to-end with the stub client
# ---------------------------------------------------------------------------


def test_semantic_audit_end_to_end_with_a_valid_response():
    client = StubClient(make_response(VALID_FINDING))
    result = semantic_audit(SOURCE, "demo.py", client=client)
    assert result.ran
    assert len(result.findings) == 1
    assert result.findings[0].line_start == 5
    # The numbered source was actually sent.
    assert "   5 | merged" in client.calls[0][1]


def test_semantic_audit_records_rejections_without_crashing():
    client = StubClient(
        make_response(dict(VALID_FINDING, line_start=400))
    )
    result = semantic_audit(SOURCE, "demo.py", client=client)
    assert result.ran
    assert result.findings == []
    assert result.rejected and any("grounding" in n for n in result.notes)


def test_semantic_audit_rejects_prose_wholesale():
    client = StubClient("Sure! Here are the issues I found:\n1. line 5 ...")
    result = semantic_audit(SOURCE, "demo.py", client=client)
    assert result.ran
    assert result.findings == []
    assert any("unparseable" in r["reason"] for r in result.rejected)


def test_semantic_audit_survives_a_client_exception():
    class FailingClient:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("rate limited")

    result = semantic_audit(SOURCE, "demo.py", client=FailingClient())
    assert not result.ran
    assert any("failed to run" in n for n in result.notes)


def test_offline_mode_returns_an_explanatory_empty_result(monkeypatch):
    monkeypatch.setenv("AUDITOR_OFFLINE", "1")
    result = semantic_audit(SOURCE, "demo.py", client=None)
    assert not result.ran
    assert result.findings == []
    assert any("skipped" in n for n in result.notes)


def test_no_key_and_no_client_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUDITOR_OFFLINE", "0")
    result = semantic_audit(SOURCE, "demo.py", client=None)
    assert not result.ran and result.findings == []


def test_definition_of_done_no_invented_line_survives_twenty_runs():
    """The B3 acceptance criterion, made executable.

    Twenty scripted responses, every one containing at least one finding with
    an invented line number. Zero of those findings may be accepted.
    """
    for seed in range(20):
        bogus = dict(VALID_FINDING, line_start=100 + seed, line_end=100 + seed)
        mixed = make_response(bogus, VALID_FINDING)
        result = semantic_audit(SOURCE, "demo.py", client=StubClient(mixed))
        assert all(
            1 <= f.line_start <= len(SOURCE.splitlines()) for f in result.findings
        )
        assert all(f.line_start != 100 + seed for f in result.findings)
