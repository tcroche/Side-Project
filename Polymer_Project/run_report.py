"""
run_report.py -- one command, one audit report.

    python run_report.py --code m2_backtester --out report.html
    python run_report.py --code m2_backtester --llm --out report.html
    python run_report.py --code m2_backtester --out report.html \\
        --trials data/trials_m2_with_rut.csv --meta data/trials_m2_meta.csv \\
        --sharpe-col sharpe_annual_with_rut --label "Universe with RUT"

The report has three sections that are never merged: statistical deflation,
deterministic AST findings, and semantic findings from the language model. Any
section that was not run says so, with the reason, because an empty section
must never read as a clean bill of health.

Output is a single self-contained HTML file: no external asset, opens offline,
and prints to A4 PDF from any browser (Ctrl+P). Nothing here needs a new
dependency.

Synthetic input is stamped, not trusted: when the metadata carries the
`is_synthetic` flag written by make_dry_run_fixture.py, the page title, the
subtitle, the section label, the provenance table and the console all say
SYNTHETIC FIXTURE, whatever --label says. A plumbing test can never be
mistaken for a real audit (rule 9).
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pandas as pd

from auditor.ast_scan import audit_file
from auditor.cache import CachingClient
from auditor.llm_pass import load_prompt, make_client_from_env, semantic_audit_file
from auditor.schema import Finding
from auditor.schema import MANUAL_CHECKLIST
from report.corroboration import annotate
from report.deflation import DeflationSection, run_deflation
from report.render import (
    LedgerEntry,
    hero_from_section,
    render_ast,
    render_deflation,
    render_page,
    render_semantic,
)

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", "data"}


def collect_paths(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        sys.exit(f"No such file or directory: {target}")
    out: list[str] = []
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py")]
    return sorted(out)


def load_trials(trials_path: str, meta_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (trials_path, meta_path):
        if not os.path.exists(path):
            sys.exit(f"Missing {path}.")
    frame = pd.read_csv(trials_path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    meta = pd.read_csv(meta_path)
    required = {"trial_id", "trial_kind", "is_frozen_cell"}
    missing = required - set(meta.columns)
    if missing:
        sys.exit(f"{meta_path} is missing required columns: {sorted(missing)}")
    meta["is_frozen_cell"] = meta["is_frozen_cell"].astype(bool)
    return frame, meta


SYNTHETIC_TAG = "SYNTHETIC FIXTURE"


def is_synthetic(meta: pd.DataFrame) -> bool:
    """True when the metadata carries the fixture flag on any row.

    make_dry_run_fixture.py stamps `is_synthetic = True`; the real export has
    no such column. Read here, in code, so that no --label can turn a plumbing
    test into something that reads like a real audit (rule 9).
    """
    if "is_synthetic" not in meta.columns:
        return False
    return bool(meta["is_synthetic"].astype(str).str.lower().isin({"true", "1"}).any())


def build_ledger(
    deflation_sections: list[DeflationSection],
    n_files: int,
    n_ast: int,
    summary,
    ran_llm: bool,
) -> list[LedgerEntry]:
    """The strip under the masthead: one cell per kind of evidence.

    Every cell states a measured figure with its threshold, or the words
    "Not run". A tier that did not run must occupy its cell rather than
    disappear from it: a missing cell reads as nothing to report, which is the
    failure mode this whole report is built against.
    """
    if deflation_sections:
        head = deflation_sections[0].headline
        measured = LedgerEntry(
            "measured", "Deflated Sharpe ratio, probability in [0,1]",
            f"{head.dsr:.3f}",
            f"{'Passes' if head.passes else 'Rejects'} at {head.confidence:.2f}",
            anchor="s01", failing=not head.passes,
        )
    else:
        measured = LedgerEntry(
            "measured", "No trial matrix supplied, so the question was not asked",
            "Not run", "Absence is not a clean bill of health", anchor="s01",
        )

    parsed = LedgerEntry(
        "parsed", f"Deterministic rules over {n_files} file(s)", f"{n_ast}",
        "Clean" if n_ast == 0 else ("1 finding" if n_ast == 1 else f"{n_ast} findings"),
        anchor="s02",
    )

    if ran_llm and summary is not None:
        model = LedgerEntry(
            "model", f"{summary.n_novel} outside the rules' reach, none verified here",
            f"{summary.n_semantic}", "To verify", anchor="s03",
        )
    else:
        model = LedgerEntry(
            "model", "Absence here is not evidence of clean code", "Not run",
            "Not asked", anchor="s03",
        )

    unseen = LedgerEntry(
        "unseen", "Properties of the data, invisible to any code reader",
        f"{len(MANUAL_CHECKLIST)}", "Open questions", anchor="s04",
    )
    return [measured, parsed, model, unseen]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full audit report.")
    parser.add_argument("--code", required=True, help="a .py file or a directory to audit")
    parser.add_argument("--out", default="audit_report.html", help="output HTML path")
    parser.add_argument("--llm", action="store_true",
                        help="run the semantic pass (needs ANTHROPIC_API_KEY; cached)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the model-response cache")
    parser.add_argument("--trials", help="CSV of trial returns (observations x trial_id)")
    parser.add_argument("--meta", help="CSV of trial metadata")
    parser.add_argument("--sharpe-col", default="sharpe_annual_with_rut",
                        help="column of annualized trial Sharpe ratios in the metadata")
    parser.add_argument("--label", default="Trial set", help="label for the deflation section")
    parser.add_argument("--title", default="Backtest Integrity Audit")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["AUDITOR_NO_CACHE"] = "1"

    # --- section 1: deflation ------------------------------------------------
    deflation_sections = []
    synthetic = False
    if args.trials and args.meta:
        frame, meta = load_trials(args.trials, args.meta)
        synthetic = is_synthetic(meta)
        label = f"{SYNTHETIC_TAG}: {args.label}" if synthetic else args.label
        deflation_sections.append(
            run_deflation(frame, meta, label=label, sharpe_col=args.sharpe_col)
        )
    elif args.trials or args.meta:
        sys.exit("--trials and --meta must be supplied together.")

    # --- section 2: deterministic --------------------------------------------
    paths = collect_paths(args.code)
    ast_findings: list[Finding] = []
    for path in paths:
        ast_findings.extend(audit_file(path))

    # --- section 3: semantic --------------------------------------------------
    semantic_findings: list[Finding] = []
    ran_llm = False
    prompt_version = ""
    n_rejected = n_capped = 0
    reason = "The semantic pass was not enabled (pass --llm to run it)."
    client = None
    if args.llm:
        client = make_client_from_env(load_prompt())
        if client is None:
            reason = ("No ANTHROPIC_API_KEY was available (or AUDITOR_OFFLINE=1), so "
                      "the semantic pass could not run.")
        else:
            for path in paths:
                result = semantic_audit_file(path, client=client)
                semantic_findings.extend(result.findings)
                n_rejected += len(result.rejected)
                n_capped += len(result.capped)
                prompt_version = result.prompt_version or prompt_version
                ran_llm = ran_llm or result.ran
            if not ran_llm:
                reason = "The semantic pass was enabled but no file completed a call."

    summary = annotate(ast_findings, semantic_findings) if ran_llm else None

    # --- assemble -------------------------------------------------------------
    provenance = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": "python " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "code audited": f"{args.code} ({len(paths)} .py file(s))",
        "deflation input": (
            (f"{args.trials} + {args.meta}"
             + (f"  [{SYNTHETIC_TAG}: is_synthetic=True in the metadata]" if synthetic else ""))
            if deflation_sections else "not supplied"
        ),
        "semantic pass": (f"prompt {prompt_version}" if ran_llm else "not run"),
        "python": platform.python_version(),
    }
    if isinstance(client, CachingClient):
        provenance["model cache"] = f"{client.hits} hit(s), {client.misses} miss(es)"

    # Plain-text separators: the renderer escapes this string, as it must,
    # so an HTML entity written here would show up literally on the page.
    subtitle = (
        (f"{SYNTHETIC_TAG}: every number on this page is meaningless "
         f"(plumbing test only)  |  " if synthetic else "")
        + f"Generated {provenance['generated']}  |  "
        f"{len(paths)} file(s) scanned  |  "
        f"{len(ast_findings)} deterministic finding(s)  |  "
        + (f"{len(semantic_findings)} semantic finding(s) to verify"
           if ran_llm else "semantic pass not run")
    )
    title = f"{SYNTHETIC_TAG} (dry run): {args.title}" if synthetic else args.title

    page = render_page(
        title=title,
        subtitle=subtitle,
        deflation_html=render_deflation(deflation_sections),
        ast_html=render_ast(ast_findings, len(paths)),
        semantic_html=render_semantic(
            summary, ran=ran_llm, prompt_version=prompt_version,
            n_rejected=n_rejected, n_capped=n_capped, reason_not_run=reason,
        ),
        provenance=provenance,
        ledger=build_ledger(deflation_sections, len(paths), len(ast_findings),
                            summary, ran_llm),
        hero=hero_from_section(deflation_sections[0] if deflation_sections else None),
    )

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(page)

    print(f"Report written to {args.out}")
    print(f"  files scanned        : {len(paths)}")
    print(f"  deterministic        : {len(ast_findings)} finding(s)")
    if ran_llm and summary is not None:
        print(f"  semantic (to verify) : {summary.n_semantic} accepted, "
              f"{n_rejected} rejected, {n_capped} capped, "
              f"{summary.n_corroborated} corroborating a rule, "
              f"{summary.n_novel} outside the rules' reach")
    else:
        print("  semantic             : not run")
    if deflation_sections:
        for s in deflation_sections:
            print(f"  deflation [{s.label}] : DSR {s.headline.dsr:.4f} "
                  f"({s.headline.verdict}), PBO {s.pbo.pbo:.3f} at percentile "
                  f"{s.pbo_percentile:.2f} of its null")
    else:
        print("  deflation            : not run (no --trials/--meta)")
    if isinstance(client, CachingClient):
        print(f"  {client.stats()}")
    if synthetic:
        print(f"  WARNING: the deflation input is a {SYNTHETIC_TAG} (is_synthetic=True); "
              f"its numbers are meaningless and the page is stamped accordingly. "
              f"Delete data\\dryrun_*.csv before any real run.")
    print("\nOpen it in a browser; Ctrl+P prints it to A4 PDF.")


if __name__ == "__main__":
    main()
