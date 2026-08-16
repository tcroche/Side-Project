"""
run_audit.py -- run the deterministic leakage rules over Python files.

From the project root:

    python run_audit.py path/to/backtest.py
    python run_audit.py path/to/repo/            # every .py file, recursively
    python run_audit.py path/to/file.py --json   # machine-readable output

The manual checklist is printed after the findings. Those are the questions a
syntax tree cannot answer; presenting them as questions rather than as
detections is the point.
"""

from __future__ import annotations

import argparse
import os
import sys

try:  # load .env from the working directory, so PyCharm runs pick up the key
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; the environment may set the key itself
    pass

from auditor.ast_scan import audit_file
from auditor.cache import CachingClient
from auditor.llm_pass import (
    SemanticAuditResult,
    load_prompt,
    make_client_from_env,
    semantic_audit_file,
)
from auditor.schema import MANUAL_CHECKLIST, Finding, findings_to_json

SEVERITY_ORDER = {"high": 0, "medium": 1, "review": 2}


def collect_paths(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        out = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv"}]
            out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py")]
        return sorted(out)
    sys.exit(f"No such file or directory: {target}")


def print_human(findings: list[Finding], paths: list[str]) -> None:
    print("BACKTEST INTEGRITY AUDITOR -- deterministic code audit")
    print(f"files scanned : {len(paths)}")
    print(f"findings      : {len(findings)}")

    if not findings:
        print("\nNo deterministic leakage pattern matched.")
        print(
            "This is a statement about the CODE only. It says nothing about how many\n"
            "configurations were tried, nor about the vintage of the data. Run the\n"
            "deflation module for the first, and read the checklist below for the second."
        )
    else:
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: SEVERITY_ORDER[kv[0]]))
        print(f"by severity   : {summary}")
        print()
        for f in sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.filename, f.line_start)):
            print("-" * 72)
            print(f"[{f.severity.upper():6}] {f.rule_id}  {f.title}")
            print(f"  {f.filename}:{f.line_start}"
                  + (f"-{f.line_end}" if f.line_end != f.line_start else ""))
            print(f"  | {f.snippet}")
            print(f"  why : {f.explanation}")
            print(f"  fix : {f.suggested_fix}")
        print("-" * 72)
        if any(f.severity == "review" for f in findings):
            print(
                "\n'review' findings are heuristics driven by naming conventions. They\n"
                "are listed separately on purpose and are not counted as detections."
            )

    print("\nWHAT THIS AUDIT CANNOT SEE")
    print("=" * 72)
    for i, item in enumerate(MANUAL_CHECKLIST, start=1):
        print(f"{i}. {item}")


def print_semantic(results: list[tuple[str, SemanticAuditResult]]) -> None:
    total = sum(len(r.findings) for _, r in results)
    rejected = sum(len(r.rejected) for _, r in results)
    ran = any(r.ran for _, r in results)

    print("\nSEMANTIC FINDINGS (LLM) -- TO VERIFY")
    print("=" * 72)
    if not ran:
        for _, r in results:
            for note in r.notes:
                print(f"  {note}")
            break
        return
    capped = sum(len(r.capped) for _, r in results)
    version = next((r.prompt_version for _, r in results if r.prompt_version), "?")
    print(f"prompt version : {version}   accepted: {total}   "
          f"rejected by grounding: {rejected}   capped at review: {capped}")
    print(
        "These findings come from a language model. Line numbers were verified\n"
        "against the file and snippets extracted from the real source, but the\n"
        "REASONING is the model's: each one is a question, not a verdict."
    )
    for path, r in results:
        for f in r.findings:
            print("-" * 72)
            print(f"[{f.severity.upper():6}] SEM  {path}:{f.line_start}"
                  + (f"-{f.line_end}" if f.line_end != f.line_start else ""))
            print(f"  | {f.snippet}")
            print(f"  why : {f.explanation}")
            if f.external_dependency:
                print(f"  check : {f.external_dependency}")
            if f.capped_from:
                reason = ("cross-file dependency declared" if f.external_dependency
                          else "external_dependency declaration missing")
                print(f"  (severity capped by the harness from "
                      f"{f.capped_from.upper()}: {reason})")
            print(f"  fix : {f.suggested_fix}")
        for note in r.notes:
            print(f"  note ({os.path.basename(path)}): {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Python code for look-ahead bias.")
    parser.add_argument("target", help="a .py file or a directory")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--llm", action="store_true",
        help="also run the semantic LLM pass (needs ANTHROPIC_API_KEY; "
             "silently degrades to AST-only when offline)",
    )
    parser.add_argument(
        "--fail-on", choices=["high", "medium", "review", "never"], default="never",
        help="exit with status 1 when a finding at this severity or above is present",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="bypass the on-disk cache of model responses (fresh API calls)",
    )
    args = parser.parse_args()

    if args.no_cache:
        os.environ["AUDITOR_NO_CACHE"] = "1"

    paths = collect_paths(args.target)
    findings: list[Finding] = []
    for path in paths:
        findings.extend(audit_file(path))

    semantic_results: list[tuple[str, SemanticAuditResult]] = []
    shared_client = None
    if args.llm:
        # One client for the whole run: the cache wrapper's hit/miss counters
        # then describe the run as a whole and can be printed at the end.
        shared_client = make_client_from_env(load_prompt())
        for path in paths:
            semantic_results.append(
                (path, semantic_audit_file(path, client=shared_client))
            )

    if args.json:
        payload = {
            "ast_findings": [f.to_dict() for f in findings],
            "semantic": {path: res.to_dict() for path, res in semantic_results},
        }
        import json as _json
        print(_json.dumps(payload, indent=2))
    else:
        print_human(findings, paths)
        if semantic_results:
            print_semantic(semantic_results)
        if isinstance(shared_client, CachingClient):
            print(f"\n{shared_client.stats()}")

    if args.fail_on != "never":
        threshold = SEVERITY_ORDER[args.fail_on]
        if any(SEVERITY_ORDER[f.severity] <= threshold for f in findings):
            sys.exit(1)


if __name__ == "__main__":
    main()
