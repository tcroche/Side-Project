"""
Semantic leakage pass, backed by a language model.

Division of labour
------------------
The AST scanner owns everything a syntax rule can express, exactly and for
free. This module owns only what a rule cannot: a `merge_asof` whose direction
makes rows see later events, a label whose definition quietly reads past the
decision time, a custom function that looks ahead without using any banned
keyword. The prompt explicitly forbids the model from re-reporting the AST
rules' territory.

Trust model
-----------
Nothing the model says is taken on faith:

  * line numbers are checked against the actual file; a finding citing a line
    that does not exist is REJECTED and counted, never shown;
  * the snippet attached to each finding is extracted from the real source by
    this code, not copied from the model's output;
  * severities outside the vocabulary are coerced to "review";
  * "high" and "medium" must be EARNED: a finding is only eligible when the
    model explicitly declares `external_dependency: null` (the leak is
    established within this file alone). A declared dependency -- or a
    missing field -- caps the severity at "review" deterministically, in this
    code, with the original severity recorded and counted. Omission is never
    a path to a higher severity. This constraint used to live in the prompt
    (v1.1.0 rule 6) and was violated on first contact with real code; a
    prompt rule is a request, this function is a guarantee;
  * temperature is 0 and the output must be JSON; anything unparseable
    rejects the whole response rather than guessing.

Findings that survive carry detector="llm" and are displayed in their own
"to verify" section, never merged with the deterministic findings.

Offline mode
------------
With AUDITOR_OFFLINE=1 or no API key, the pass returns an empty result with an
explanatory note, so the rest of the tool works with no network and no account.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import yaml

from auditor.schema import Finding

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "code_auditor_v1.yaml")
VALID_SEVERITIES = {"high", "medium", "review"}
MAX_FINDINGS_PER_FILE = 50


# ---------------------------------------------------------------------------
# Client abstraction, so tests can inject a stub and never touch the network
# ---------------------------------------------------------------------------


class CompletionClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Thin wrapper over the Anthropic API. Constructed only when a key exists."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        import anthropic  # imported lazily so offline mode needs no package

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            system=system,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )


def load_prompt(path: str = PROMPT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    for key in ("system_prompt", "model", "temperature", "version"):
        if key not in spec:
            raise ValueError(f"Prompt file {path} is missing '{key}'.")
    return spec


def make_client_from_env(spec: dict) -> CompletionClient | None:
    """Return a real client, or None when running offline.

    Unless AUDITOR_NO_CACHE=1, the client is wrapped in a CachingClient keyed
    on the sha256 of the exact (system, user) pair, so a prompt bump or a
    source change can never be served a stale response, while re-runs on
    unchanged inputs cost nothing. Cache directory: AUDITOR_CACHE_DIR or
    data/llm_cache under the working directory.
    """
    if os.environ.get("AUDITOR_OFFLINE", "0") == "1":
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    model = os.environ.get("AUDITOR_MODEL", spec["model"])
    client: CompletionClient = AnthropicClient(
        model=model,
        temperature=float(spec.get("temperature", 0.0)),
        max_tokens=int(spec.get("max_tokens", 2048)),
    )
    if os.environ.get("AUDITOR_NO_CACHE", "0") != "1":
        from auditor.cache import CachingClient

        cache_dir = os.environ.get(
            "AUDITOR_CACHE_DIR", os.path.join("data", "llm_cache")
        )
        client = CachingClient(client, cache_dir)
    return client


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SemanticAuditResult:
    findings: list[Finding] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {reason, raw}
    capped: list[dict] = field(default_factory=list)  # {line_start, capped_from, ...}
    notes: list[str] = field(default_factory=list)
    prompt_version: str = ""
    ran: bool = False

    def to_dict(self) -> dict:
        return {
            "ran": self.ran,
            "prompt_version": self.prompt_version,
            "findings": [f.to_dict() for f in self.findings],
            "rejected": self.rejected,
            "capped": self.capped,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def number_source(source: str) -> str:
    """'  12 | code' numbering, matching what the prompt announces."""
    return "\n".join(
        f"{i:>4} | {line}" for i, line in enumerate(source.splitlines(), start=1)
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(text: str) -> dict:
    """Parse the model's output into a dict, tolerating markdown fences only.

    Anything else non-JSON is an error: with temperature 0 and a JSON-only
    instruction, prose in the output is a signal to reject, not to repair.
    """
    cleaned = _FENCE.sub("", text.strip()).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(cleaned[start : end + 1])


_NO_DEPENDENCY_TOKENS = {"", "none", "null", "n/a", "na", "-"}


def normalize_external_dependency(raw: dict) -> tuple[str | None, bool]:
    """Read the finding's external_dependency field into (dependency, declared_self_contained).

    Three states, resolved deterministically:
      * field present and null-ish  -> (None, True):  the model declared the
        leak established within this file alone; high/medium are reachable.
      * field present, non-empty str -> (text, False): a cross-file dependency
        is declared; the harness will cap the severity.
      * field missing or wrong type  -> (None, False): undeclared. Treated
        exactly like a declared dependency, so omission never buys severity.
    """
    if "external_dependency" not in raw:
        return None, False
    value = raw.get("external_dependency")
    if value is None:
        return None, True
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NO_DEPENDENCY_TOKENS:
            return None, True
        return text, False
    return None, False  # wrong type: undeclared


def ground_findings(
    payload: dict,
    source: str,
    filename: str,
    *,
    enforce_external_cap: bool = True,
) -> tuple[list[Finding], list[dict]]:
    """Validate the model's findings against the actual file.

    This is the guarantee the design leans on: grounding is verified by CODE,
    not asserted by the model. Anything that fails is returned in `rejected`
    with a reason, so the failure rate is measurable.

    `enforce_external_cap=False` exists ONLY for the benchmark ablation
    (same cached model outputs, post-processed with and without the cap).
    Every user-facing path keeps the default.
    """
    lines = source.splitlines()
    n_lines = len(lines)

    accepted: list[Finding] = []
    rejected: list[dict] = []

    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        return [], [{"reason": "'findings' is not a list", "raw": payload}]

    for raw in raw_findings[:MAX_FINDINGS_PER_FILE]:
        if not isinstance(raw, dict):
            rejected.append({"reason": "finding is not an object", "raw": raw})
            continue

        try:
            line_start = int(raw.get("line_start"))
            line_end = int(raw.get("line_end", line_start))
        except (TypeError, ValueError):
            rejected.append({"reason": "non-integer line numbers", "raw": raw})
            continue

        if not (1 <= line_start <= n_lines):
            rejected.append(
                {"reason": f"line_start {line_start} outside 1..{n_lines}", "raw": raw}
            )
            continue
        if not (line_start <= line_end <= n_lines):
            rejected.append(
                {"reason": f"line_end {line_end} invalid for file of {n_lines} lines",
                 "raw": raw}
            )
            continue

        severity = str(raw.get("severity", "")).lower()
        if severity not in VALID_SEVERITIES:
            severity = "review"

        explanation = str(raw.get("explanation", "")).strip()
        if not explanation:
            rejected.append({"reason": "empty explanation", "raw": raw})
            continue

        # Severity entitlement is decided HERE, not believed from the model.
        # A declared cross-file dependency -- or a missing declaration -- caps
        # the finding at "review"; the original claim is kept for the counter.
        dependency, self_contained = normalize_external_dependency(raw)
        capped_from: str | None = None
        if enforce_external_cap and not self_contained and severity in ("high", "medium"):
            capped_from = severity
            severity = "review"

        # The snippet comes from the REAL file, never from the model.
        snippet = lines[line_start - 1].strip()[:160]

        accepted.append(
            Finding(
                rule_id="SEM",
                title="Semantic leakage (LLM, to verify)",
                severity=severity,
                line_start=line_start,
                line_end=line_end,
                snippet=snippet,
                explanation=explanation,
                suggested_fix=str(raw.get("suggested_fix", "")).strip()
                or "Review the flagged lines and re-derive what is knowable at decision time.",
                detector="llm",
                filename=filename,
                external_dependency=dependency,
                capped_from=capped_from,
            )
        )

    return accepted, rejected


def semantic_audit(
    source: str,
    filename: str = "<string>",
    *,
    client: CompletionClient | None = None,
    prompt_path: str = PROMPT_PATH,
    enforce_external_cap: bool = True,
) -> SemanticAuditResult:
    """Run the semantic pass on one file.

    `client=None` triggers environment-based construction; if that also yields
    nothing (offline mode or no key), the result explains itself and the caller
    can proceed with the deterministic findings alone.
    """
    spec = load_prompt(prompt_path)
    result = SemanticAuditResult(prompt_version=str(spec["version"]))

    if client is None:
        client = make_client_from_env(spec)
    if client is None:
        result.notes.append(
            "Semantic pass skipped: no ANTHROPIC_API_KEY (or AUDITOR_OFFLINE=1). "
            "Deterministic findings are unaffected."
        )
        return result

    user_message = (
        f"File: {os.path.basename(filename)}\n"
        f"{number_source(source)}"
    )

    try:
        raw_text = client.complete(spec["system_prompt"], user_message)
    except Exception as exc:  # network, auth, rate limit -- degrade, don't crash
        result.notes.append(f"Semantic pass failed to run: {exc}")
        return result

    result.ran = True

    try:
        payload = parse_model_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        result.rejected.append({"reason": f"unparseable output: {exc}", "raw": raw_text[:500]})
        result.notes.append(
            "Model output was not valid JSON; the whole response was rejected "
            "rather than repaired."
        )
        return result

    result.findings, result.rejected = ground_findings(
        payload, source, filename, enforce_external_cap=enforce_external_cap
    )
    result.capped = [
        {
            "filename": f.filename,
            "line_start": f.line_start,
            "capped_from": f.capped_from,
            "external_dependency": f.external_dependency,
        }
        for f in result.findings
        if f.capped_from is not None
    ]
    if result.capped:
        result.notes.append(
            f"{len(result.capped)} finding(s) capped at 'review' by the harness "
            f"(cross-file dependency declared, or declaration missing); the "
            f"original severities are recorded in the JSON output."
        )
    if result.rejected:
        result.notes.append(
            f"{len(result.rejected)} finding(s) rejected by grounding checks; "
            f"rejection reasons are recorded in the JSON output."
        )
    return result


def semantic_audit_file(
    path: str,
    *,
    client: CompletionClient | None = None,
    enforce_external_cap: bool = True,
) -> SemanticAuditResult:
    with open(path, "r", encoding="utf-8") as handle:
        return semantic_audit(
            handle.read(),
            filename=path,
            client=client,
            enforce_external_cap=enforce_external_cap,
        )


__all__ = [
    "AnthropicClient",
    "CompletionClient",
    "SemanticAuditResult",
    "semantic_audit",
    "semantic_audit_file",
    "number_source",
    "parse_model_json",
    "ground_findings",
    "normalize_external_dependency",
    "load_prompt",
    "make_client_from_env",
]
