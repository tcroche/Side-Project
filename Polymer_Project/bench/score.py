"""
Scoring for the seeded-bug benchmark.

Two decisions define every number this module produces, and both are stated
here rather than buried in code:

1. DETECTION THRESHOLD. A finding counts as a detection only at severity
   "high" or "medium". This mirrors the tool's own reporting convention
   ("'review' findings ... are not counted as detections"): review-level items
   are questions, and a question is neither a hit nor a false alarm. They are
   counted separately as `review_questions`.

2. LOCALISATION MATCHING. A detection is a true positive when its line range
   overlaps a seeded leak in the same file; recall asks whether each seeded
   leak was overlapped by at least one detection. Matching by lines rather
   than by rule identity keeps the metric detector-agnostic (the LLM has no
   rule ids) and keeps calibration (severity) a SEPARATE question from
   localisation -- the severity cap must never masquerade as lost recall.

Precision is computed over every detection the detector emitted across ALL
case files, clean ones included. Multiple detections overlapping the same leak
are all true positives (pointing twice at a real leak is not a false alarm).
A true positive whose matched leak belongs to the *other* detector's territory
is still a true positive, but it is tallied so the runner can report the
LLM's out-of-lane hits (the prompt forbids re-reporting catalogue patterns).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from auditor.schema import Finding
from bench.truth import Case, Leak

DETECTION_SEVERITIES = {"high", "medium"}


def is_detection(finding: Finding) -> bool:
    return finding.severity in DETECTION_SEVERITIES


def overlaps(finding: Finding, leak: Leak) -> bool:
    return finding.line_start <= leak.line_end and leak.line_start <= finding.line_end


@dataclass
class DetectorScore:
    name: str
    n_detections: int = 0
    tp: int = 0
    fp: int = 0
    tp_catalogue: int = 0            # TPs matched to catalogue-family leaks
    tp_semantic: int = 0             # TPs matched to semantic-family leaks
    leaks_found_catalogue: int = 0
    leaks_found_semantic: int = 0
    leaks_total_catalogue: int = 0
    leaks_total_semantic: int = 0
    clean_false_positives: int = 0   # detections on clean/dependent files
    review_questions: int = 0        # review-level findings, any file (not scored)
    per_case: dict = field(default_factory=dict)

    # -- derived metrics -----------------------------------------------------

    @property
    def precision(self) -> float:
        return self.tp / self.n_detections if self.n_detections else 0.0

    @property
    def recall_catalogue(self) -> float:
        total = self.leaks_total_catalogue
        return self.leaks_found_catalogue / total if total else 0.0

    @property
    def recall_semantic(self) -> float:
        total = self.leaks_total_semantic
        return self.leaks_found_semantic / total if total else 0.0

    @property
    def recall_overall(self) -> float:
        found = self.leaks_found_catalogue + self.leaks_found_semantic
        total = self.leaks_total_catalogue + self.leaks_total_semantic
        return found / total if total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall_overall
        return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_detections": self.n_detections,
            "tp": self.tp,
            "fp": self.fp,
            "tp_catalogue": self.tp_catalogue,
            "tp_semantic": self.tp_semantic,
            "precision": self.precision,
            "recall_catalogue": self.recall_catalogue,
            "recall_semantic": self.recall_semantic,
            "recall_overall": self.recall_overall,
            "f1": self.f1,
            "clean_false_positives": self.clean_false_positives,
            "review_questions": self.review_questions,
            "leaks_total_catalogue": self.leaks_total_catalogue,
            "leaks_total_semantic": self.leaks_total_semantic,
            "per_case": self.per_case,
        }


def score_detector(
    name: str,
    findings_by_case: dict[str, list[Finding]],
    cases: tuple[Case, ...],
) -> DetectorScore:
    """Score one detector's findings against the seeded truth.

    `findings_by_case` maps case FILENAMES (basenames) to that detector's
    findings for the file; missing keys mean the detector saw nothing there.
    """
    score = DetectorScore(name=name)

    for case in cases:
        findings = findings_by_case.get(case.filename, [])
        detections = [f for f in findings if is_detection(f)]
        score.review_questions += sum(1 for f in findings if f.severity == "review")
        score.n_detections += len(detections)

        for leak in case.leaks:
            if leak.family == "catalogue":
                score.leaks_total_catalogue += 1
            else:
                score.leaks_total_semantic += 1

        case_tp = case_fp = 0
        for detection in detections:
            matched = [leak for leak in case.leaks if overlaps(detection, leak)]
            if matched:
                score.tp += 1
                case_tp += 1
                if any(leak.family == "catalogue" for leak in matched):
                    score.tp_catalogue += 1
                if any(leak.family == "semantic" for leak in matched):
                    score.tp_semantic += 1
            else:
                score.fp += 1
                case_fp += 1
                if case.kind in ("clean", "dependent"):
                    score.clean_false_positives += 1

        for leak in case.leaks:
            found = any(overlaps(d, leak) for d in detections)
            if found and leak.family == "catalogue":
                score.leaks_found_catalogue += 1
            if found and leak.family == "semantic":
                score.leaks_found_semantic += 1

        score.per_case[case.filename] = {
            "kind": case.kind,
            "detections": len(detections),
            "tp": case_tp,
            "fp": case_fp,
        }

    return score


def merge_findings(
    *sources: dict[str, list[Finding]],
) -> dict[str, list[Finding]]:
    """Union of detectors' findings per case, for the hybrid row. No
    deduplication: two detectors pointing at the same real leak are two true
    positives, which is the honest reading of agreement."""
    merged: dict[str, list[Finding]] = {}
    for source in sources:
        for filename, findings in source.items():
            merged.setdefault(filename, []).extend(findings)
    return merged


__all__ = [
    "DETECTION_SEVERITIES",
    "DetectorScore",
    "is_detection",
    "merge_findings",
    "overlaps",
    "score_detector",
]
