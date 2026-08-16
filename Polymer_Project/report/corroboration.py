"""
Corroboration between the two detectors, WITHOUT merging them.

Principle 4 of this project says the deterministic and semantic findings are
kept architecturally separate at every layer. That stays true here: this module
never modifies, removes or reorders an AST finding, and never promotes a
semantic finding into the deterministic section. It only computes an
ANNOTATION, attached to the semantic side, saying "a rule already found
something on these lines".

Why it is worth computing. The benchmark measured that the model re-reports
patterns the prompt told it to leave to the rules (exclusion compliance 2/8).
Rather than pretend that instruction works, the report states the overlap and
counts it: of N semantic findings, how many restate a rule, and how many are
genuinely outside the rules' reach. The second number is the semantic pass's
actual contribution, and it is the honest thing to show a reader.

Overlap is decided on line ranges within the same file, the same rule the
benchmark scorer uses, so the report and the benchmark cannot disagree about
what "the same finding" means.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from auditor.schema import Finding


def _normalise(path: str) -> str:
    """Path with separators unified, lowercased.

    `os.path` only treats the HOST separator as one: on Linux,
    `basename("m2_backtester\\strategy.py")` returns the whole string. The two
    detectors are handed the same file by the same runner, so in practice the
    spelling matches, but the audited paths come from the user's command line
    and the report must not depend on which OS produced them. Caught by a test
    running on Linux against Windows-style paths.
    """
    return os.path.normcase(path.replace("\\", "/").strip("/"))


def _same_file(a: str, b: str) -> bool:
    """Compare paths tolerantly: the two detectors may be handed the same file
    with different separators or as absolute vs relative paths."""
    if not a or not b:
        return False
    na, nb = _normalise(a), _normalise(b)
    if na == nb:
        return True
    return na.rsplit("/", 1)[-1] == nb.rsplit("/", 1)[-1]


def _overlaps(a: Finding, b: Finding) -> bool:
    return a.line_start <= b.line_end and b.line_start <= a.line_end


@dataclass(frozen=True)
class AnnotatedSemanticFinding:
    """A semantic finding plus what the deterministic layer says about it.

    The finding itself is untouched; `corroborates` is metadata computed at
    reporting time.
    """

    finding: Finding
    corroborates: tuple[tuple[str, int], ...] = ()  # (rule_id, line_start)

    @property
    def is_corroborated(self) -> bool:
        return bool(self.corroborates)

    @property
    def label(self) -> str:
        """Human-readable annotation, or an empty string when novel."""
        if not self.corroborates:
            return ""
        parts = ", ".join(f"{rule} at line {line}" for rule, line in self.corroborates)
        return f"corroborates {parts}"


@dataclass(frozen=True)
class CorroborationSummary:
    annotated: tuple[AnnotatedSemanticFinding, ...]
    n_semantic: int
    n_corroborated: int
    n_novel: int

    @property
    def novel_findings(self) -> tuple[AnnotatedSemanticFinding, ...]:
        return tuple(a for a in self.annotated if not a.is_corroborated)


def annotate(
    ast_findings: list[Finding],
    semantic_findings: list[Finding],
) -> CorroborationSummary:
    """Annotate each semantic finding with the AST findings it overlaps.

    `ast_findings` is read only. The returned objects wrap the semantic
    findings; the originals are unmodified frozen dataclasses.
    """
    annotated: list[AnnotatedSemanticFinding] = []
    for sem in semantic_findings:
        hits = tuple(
            sorted(
                {
                    (ast.rule_id, ast.line_start)
                    for ast in ast_findings
                    if _same_file(ast.filename, sem.filename) and _overlaps(ast, sem)
                }
            )
        )
        annotated.append(AnnotatedSemanticFinding(finding=sem, corroborates=hits))

    n_corroborated = sum(1 for a in annotated if a.is_corroborated)
    return CorroborationSummary(
        annotated=tuple(annotated),
        n_semantic=len(annotated),
        n_corroborated=n_corroborated,
        n_novel=len(annotated) - n_corroborated,
    )


__all__ = [
    "AnnotatedSemanticFinding",
    "CorroborationSummary",
    "annotate",
]
