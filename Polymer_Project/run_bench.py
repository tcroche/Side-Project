"""
run_bench.py -- the seeded-bug benchmark, end to end.

From the project root:

    python run_bench.py                     # AST only: free, offline, the dry run
    python run_bench.py --llm               # + semantic pass (cached) + ablation
    python run_bench.py --llm --json bench_results.json

16 cases in bench/cases/: 8 trapped (9 seeded catalogue leaks), 5 clean, 2
semantic (leaks no syntactic rule can express), 1 dependent (the correct
answer is a question). Ground truth lives in bench/truth.py and is re-verified
by tests/test_bench.py, so the numbers printed here can never drift from the
detectors silently.

Scoring convention (bench/score.py): detections are findings at severity
high or medium; review-level findings are questions and are counted apart.
Matching is by line overlap, so localisation and calibration stay separate.

The ablation replays the SAME cached model outputs with the severity cap
disabled: one set of API calls, two post-processings, measurable delta.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from auditor.ast_scan import audit_file
from auditor.cache import CachingClient
from auditor.llm_pass import load_prompt, make_client_from_env, semantic_audit_file
from bench.score import (
    DetectorScore,
    is_detection,
    merge_findings,
    score_detector,
)
from bench.truth import CASES, case_path, total_leaks

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_SOURCE_GLOBS = ("core/*.py", "auditor/*.py", "report/*.py",
                     "bench/*.py", "tests/*.py", "*.py")


# ---------------------------------------------------------------------------
# Detector passes
# ---------------------------------------------------------------------------


def run_ast_pass() -> dict[str, list]:
    return {case.filename: audit_file(case_path(case.filename)) for case in CASES}


def run_llm_pass(client, *, enforce_external_cap: bool):
    """Semantic pass over every case. Returns (findings_by_case, meta)."""
    findings_by_case: dict[str, list] = {}
    meta = {
        "prompt_version": "",
        "rejected": 0,
        "capped": 0,
        "notes": [],
        "results": {},
        "enforce_external_cap": enforce_external_cap,
    }
    for case in CASES:
        result = semantic_audit_file(
            case_path(case.filename),
            client=client,
            enforce_external_cap=enforce_external_cap,
        )
        findings_by_case[case.filename] = list(result.findings)
        meta["prompt_version"] = result.prompt_version or meta["prompt_version"]
        meta["rejected"] += len(result.rejected)
        meta["capped"] += len(result.capped)
        meta["notes"].extend(f"{case.filename}: {n}" for n in result.notes)
        meta["results"][case.filename] = result.to_dict()
    return findings_by_case, meta


def compare_for_ablation(capped_on: dict, capped_off: dict) -> dict:
    """Same cached outputs, two post-processings. Localisation must be
    identical by construction; the interesting numbers are the severity
    changes and the detections that only exist without the cap."""
    def locations(findings_by_case):
        return {
            (filename, f.line_start, f.line_end)
            for filename, findings in findings_by_case.items()
            for f in findings
        }

    loc_on, loc_off = locations(capped_on), locations(capped_off)

    severity_changes = 0
    for filename in capped_on:
        for f_on, f_off in zip(capped_on[filename], capped_off.get(filename, [])):
            if f_on.severity != f_off.severity:
                severity_changes += 1

    kinds = {case.filename: case.kind for case in CASES}

    def control_detections(findings_by_case):
        return sum(
            1
            for filename, findings in findings_by_case.items()
            if kinds.get(filename) in ("clean", "dependent")
            for f in findings
            if is_detection(f)
        )

    return {
        "localisation_identical": loc_on == loc_off,
        "n_locations": len(loc_on),
        "severity_changes": severity_changes,
        "control_detections_cap_on": control_detections(capped_on),
        "control_detections_cap_off": control_detections(capped_off),
    }


def audit_real_code() -> list[tuple[str, int, int]]:
    """AST pass over the tool's own source and, when present, the M2
    backtester -- live, so the printed numbers are measured, never quoted."""
    rows = []
    own = sorted(
        path
        for pattern in REPO_SOURCE_GLOBS
        for path in glob.glob(os.path.join(REPO_ROOT, pattern))
    )
    own_findings = sum(len(audit_file(p)) for p in own)
    rows.append(("this repository (self-audit)", len(own), own_findings))

    m2_dir = os.path.join(REPO_ROOT, "m2_backtester")
    if os.path.isdir(m2_dir):
        m2 = sorted(glob.glob(os.path.join(m2_dir, "**", "*.py"), recursive=True))
        m2_findings = sum(len(audit_file(p)) for p in m2)
        rows.append(("m2_backtester/ (real strategy code)", len(m2), m2_findings))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_score_table(scores: list[DetectorScore]) -> None:
    n_cat, n_sem = total_leaks("catalogue"), total_leaks("semantic")
    print("\nDetection threshold: severity >= medium "
          "('review' findings are questions, not detections)")
    header = (f"{'Detector':14}{'det':>5}{'TP':>5}{'FP':>5}{'P':>8}"
              f"{f'R cat/{n_cat}':>10}{f'R sem/{n_sem}':>10}{'R all':>8}{'F1':>8}")
    print(header)
    print("-" * len(header))
    for s in scores:
        print(f"{s.name:14}{s.n_detections:>5}{s.tp:>5}{s.fp:>5}"
              f"{s.precision:>8.2f}{s.recall_catalogue:>10.2f}"
              f"{s.recall_semantic:>10.2f}{s.recall_overall:>8.2f}{s.f1:>8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded-bug benchmark.")
    parser.add_argument("--llm", action="store_true",
                        help="also run the semantic pass and its ablation "
                             "(needs ANTHROPIC_API_KEY; responses are cached)")
    parser.add_argument("--json", metavar="PATH",
                        help="write the full results as JSON to PATH")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the response cache (fresh API calls)")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["AUDITOR_NO_CACHE"] = "1"

    n_cases = len(CASES)
    kinds = [case.kind for case in CASES]
    print("SEEDED-BUG BENCHMARK")
    print(f"cases : {n_cases}  "
          f"(trapped {kinds.count('trapped')} / clean {kinds.count('clean')} / "
          f"semantic {kinds.count('semantic')} / dependent {kinds.count('dependent')})")
    print(f"seeded leaks : {total_leaks()}  "
          f"(catalogue {total_leaks('catalogue')}, semantic {total_leaks('semantic')})")

    # --- deterministic half -------------------------------------------------
    ast_findings = run_ast_pass()
    scores = [score_detector("AST only", ast_findings, CASES)]

    payload: dict = {"cases": n_cases, "leaks": total_leaks()}
    llm_meta = None
    ablation = None
    client = None

    # --- semantic half ------------------------------------------------------
    if args.llm:
        client = make_client_from_env(load_prompt())
        if client is None:
            print("\nSemantic pass skipped: no ANTHROPIC_API_KEY (or "
                  "AUDITOR_OFFLINE=1). AST results above are unaffected.")
        else:
            print(f"\nSemantic pass over {n_cases} files "
                  f"(identical inputs are served from the cache)...")
            llm_findings, llm_meta = run_llm_pass(client, enforce_external_cap=True)
            llm_off, _ = run_llm_pass(client, enforce_external_cap=False)
            ablation = compare_for_ablation(llm_findings, llm_off)

            scores.append(score_detector("LLM only", llm_findings, CASES))
            scores.append(
                score_detector(
                    "Hybrid", merge_findings(ast_findings, llm_findings), CASES
                )
            )
            scores.append(score_detector("LLM (cap off)", llm_off, CASES))

    print_score_table(scores)

    clean_line = "  ".join(
        f"{s.name}: {s.clean_false_positives}" for s in scores
    )
    print(f"\nFalse positives on the control files (5 clean + 1 dependent): {clean_line}")
    review_line = "  ".join(f"{s.name}: {s.review_questions}" for s in scores)
    print(f"Review-level questions raised (not scored): {review_line}")

    if llm_meta is not None:
        lane = next((s.tp_catalogue for s in scores if s.name == "LLM only"), 0)
        print(f"\nLLM plumbing -- prompt {llm_meta['prompt_version']}: "
              f"grounding rejections {llm_meta['rejected']}, "
              f"harness caps {llm_meta['capped']}, "
              f"out-of-lane catalogue hits {lane} "
              f"(the prompt forbids re-reporting catalogue patterns)")

    if ablation is not None:
        print("\n=== ABLATION: same cached model outputs, cap ON vs cap OFF ===")
        print(f"localisation identical : {ablation['localisation_identical']} "
              f"({ablation['n_locations']} finding location(s) in both runs)")
        print(f"severities changed by the cap : {ablation['severity_changes']}")
        print(f"detections on control files : cap ON {ablation['control_detections_cap_on']}"
              f" / cap OFF {ablation['control_detections_cap_off']}  "
              f"(the difference is what the cap converted into questions)")

    print("\n=== MEASURED ON REAL CODE (live, deterministic rules) ===")
    for name, n_files, n_findings in audit_real_code():
        print(f"{name:38} {n_files:>3} files, {n_findings} finding(s)")

    if isinstance(client, CachingClient):
        print(f"\n{client.stats()}")

    if args.json:
        payload["scores"] = {s.name: s.to_dict() for s in scores}
        payload["llm_meta"] = llm_meta
        payload["ablation"] = ablation
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
