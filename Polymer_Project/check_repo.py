"""
check_repo.py -- what a `git add .` would publish, checked before it happens.

    python check_repo.py            # human-readable table, exit 1 on any failure
    python check_repo.py --json     # same result as JSON

Why this exists rather than a checklist in the README: a checklist is a request
and a script is a guarantee, which is the same rule this project applies to the
language model. An API key was pasted into a conversation once on this project
and had to be revoked; the trial CSVs are derived from course-provided data that
may not be redistributed; the dry-run fixture must never leave the machine. None
of that can depend on remembering.

What it inspects is not the working tree but THE SET OF FILES GIT WOULD PUBLISH:
tracked files plus untracked files that no ignore rule covers. A 2 GB pickle
sitting in an ignored folder is fine and is reported as such; the same pickle one
directory higher is a failure.

Checks, each independently reported:

    secrets        no API key or token in any publishable file
    env            .env is not publishable
    data           no dataset, trial CSV or model cache is publishable
    fixture        no dryrun_* synthetic file exists anywhere in the tree
    generated      no generated report at the repository root is publishable
    ignore-rules   .gitignore actually covers the patterns above
    paths          no absolute machine path (C:\\Users\\..., /home/...) in the text
    writeup        the one-page PDF is present and is publishable
    license        a LICENSE file is present and is publishable
    size           nothing unexpectedly large

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))

#: Anything that looks like a live credential. The placeholder in .env.example
#: (`sk-ant-...`) deliberately does not match: the pattern requires real payload.
SECRET_PATTERNS = (
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{12,}")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

#: Machine paths leak the author's username and break for everyone else.
# A Windows path is written escaped ("C:\\Users\\...") as often as raw, so the
# separator has to tolerate one backslash or two. Caught by a test that plants
# the escaped form, which the single-separator version missed.
PATH_PATTERNS = (
    re.compile(r"[Cc]:[\\/]{1,2}Users[\\/]{1,2}[^\s\"'<>|]+"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
)

#: Files whose whole purpose is to show these shapes.
PATH_EXEMPT = {"check_repo.py", "DEVLOG.md", ".env.example"}

DATA_SUFFIXES = (".pkl", ".pickle", ".parquet", ".feather", ".h5", ".xlsx")

#: Skipped when reading text. Everything else is read and sniffed, because a
#: credential does not need a .py to sit in: LICENSE, Dockerfile and a file
#: called `key` have no extension at all, and an allow-list of extensions would
#: have left all three unscanned.
BINARY_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".ttf",
                   ".otf", ".ico", ".woff", ".woff2", ".so", ".dll", ".exe",
                   ".pyc") + DATA_SUFFIXES
REQUIRED_IGNORES = ("data/*.csv", "data/llm_cache/", ".env", "/*.html")
WRITEUP_DIR = "docs"
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
LARGE_FILE_BYTES = 2_000_000


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    offenders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"check": self.name, "ok": self.ok, "detail": self.detail,
                "offenders": self.offenders}


# ---------------------------------------------------------------------------
# What would be published
# ---------------------------------------------------------------------------


def _git(args: list[str], root: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", "-C", root] + args, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [line for line in out.stdout.splitlines() if line.strip()]


def publishable_files(root: str = ROOT) -> tuple[list[str], bool]:
    """(paths relative to root, whether git answered).

    Tracked files plus untracked files no ignore rule covers: exactly what the
    next `git add .` would stage. Without git, fall back to walking the tree and
    say so, because the fallback cannot read .gitignore and will over-report.
    """
    tracked = _git(["ls-files"], root)
    if tracked is not None:
        others = _git(["ls-files", "--others", "--exclude-standard"], root) or []
        return sorted(set(tracked) | set(others)), True

    skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".idea", ".vscode"}
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            path = os.path.relpath(os.path.join(base, name), root)
            found.append(path.replace(os.sep, "/"))
    return sorted(found), False


def _read_text(root: str, rel: str) -> str | None:
    """Text of a publishable file, or None when it is binary or too large."""
    path = os.path.join(root, rel)
    if not os.path.isfile(path) or rel.lower().endswith(BINARY_SUFFIXES):
        return None
    if os.path.getsize(path) > 3_000_000:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:            # binary without a telling extension
        return None
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_secrets(files: list[str], root: str) -> Check:
    hits = []
    for rel in files:
        text = _read_text(root, rel)
        if text is None:
            continue
        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                line = text[:match.start()].count("\n") + 1
                hits.append(f"{rel}:{line} looks like a(n) {label}")
    return Check("secrets", not hits,
                 "no credential in any publishable file" if not hits
                 else "REVOKE THE KEY FIRST, then remove it from the file and from "
                      "git history", hits)


def check_env(files: list[str], root: str) -> Check:
    hits = [f for f in files if os.path.basename(f) == ".env"]
    return Check("env", not hits,
                 ".env is not publishable" if not hits else
                 "add .env to .gitignore and untrack it", hits)


def check_data(files: list[str], root: str) -> Check:
    hits = []
    for rel in files:
        low = rel.lower()
        if low.endswith(DATA_SUFFIXES):
            hits.append(f"{rel} (dataset)")
        elif low.startswith("data/") and low.endswith(".csv"):
            hits.append(f"{rel} (trial export, derived from course data)")
        elif "llm_cache/" in low:
            hits.append(f"{rel} (raw model responses)")
    return Check("data", not hits,
                 "no dataset, trial export or model cache is publishable" if not hits
                 else "these are derived from data that is not redistributable, or "
                      "are machine-local artefacts", hits)


def check_fixture(files: list[str], root: str) -> Check:
    """Rule 9: no synthetic number may survive anywhere near a real report.

    This one looks at the whole tree, not only at what is publishable: a fixture
    left in data/ is ignored by git but can still be picked up by a command.
    """
    hits = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "venv",
                                                "__pycache__", ".pytest_cache"}]
        for name in names:
            if name.startswith("dryrun_"):
                rel = os.path.relpath(os.path.join(base, name), root)
                hits.append(rel.replace(os.sep, "/"))
    return Check("fixture", not hits,
                 "no synthetic fixture left in the tree" if not hits
                 else "delete these before any real run (rule 9)", hits)


def check_generated(files: list[str], root: str) -> Check:
    hits = [f for f in files if "/" not in f and f.endswith(".html")]
    return Check("generated", not hits,
                 "no generated report at the root is publishable" if not hits
                 else "reports belong in docs/ if you want to publish one; the root "
                      "is where run_report.py writes by default", hits)


def check_ignore_rules(files: list[str], root: str) -> Check:
    text = _read_text(root, ".gitignore") or ""
    missing = [pattern for pattern in REQUIRED_IGNORES if pattern not in text]
    return Check("ignore-rules", not missing,
                 ".gitignore covers keys, data, cache and generated reports"
                 if not missing else "add these lines to .gitignore",
                 missing)


def check_paths(files: list[str], root: str) -> Check:
    hits = []
    for rel in files:
        if os.path.basename(rel) in PATH_EXEMPT:
            continue
        text = _read_text(root, rel)
        if text is None:
            continue
        for pattern in PATH_PATTERNS:
            match = pattern.search(text)
            if match:
                line = text[:match.start()].count("\n") + 1
                hits.append(f"{rel}:{line} {match.group(0)}")
                break
    return Check("paths", not hits,
                 "no absolute machine path in the published text" if not hits
                 else "these leak a username and break on any other machine", hits)


def check_writeup(files: list[str], root: str) -> Check:
    pdfs = [f for f in files if f.startswith(f"{WRITEUP_DIR}/") and f.endswith(".pdf")]
    return Check("writeup", bool(pdfs),
                 f"the one-page write-up is published ({', '.join(pdfs)})" if pdfs
                 else f"no PDF under {WRITEUP_DIR}/: build it with "
                      f"`python {WRITEUP_DIR}/build_writeup.py`, which checks its own "
                      f"one-page and 11 pt constraints", [])


def check_license(files: list[str], root: str) -> Check:
    """Without a licence file, published code is all rights reserved by default,
    which is a strange signal on a repository one is invited to read."""
    hits = [f for f in files if f in LICENSE_NAMES]
    if not hits:
        return Check("license", False,
                     f"no licence file: add one of {', '.join(LICENSE_NAMES)}, or "
                     "the code is all rights reserved by default", [])
    text = _read_text(root, hits[0]) or ""
    if len(text.strip()) < 200:
        return Check("license", False, f"{hits[0]} looks empty or truncated", hits)
    return Check("license", True, f"published under {hits[0]}", [])


def check_size(files: list[str], root: str) -> Check:
    big = []
    total = 0
    for rel in files:
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            total += size
            if size > LARGE_FILE_BYTES:
                big.append(f"{rel} ({size / 1_000_000:.1f} MB)")
    return Check("size", not big,
                 f"{len(files)} file(s), {total / 1_000_000:.1f} MB in total"
                 if not big else "unexpectedly large file(s) about to be published",
                 big)


CHECKS = (check_secrets, check_env, check_data, check_fixture, check_generated,
          check_ignore_rules, check_paths, check_writeup, check_license, check_size)


def run_all(root: str = ROOT) -> tuple[list[Check], bool]:
    files, from_git = publishable_files(root)
    return [check(files, root) for check in CHECKS], from_git


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-publication check.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    checks, from_git = run_all(args.root)
    failed = [c for c in checks if not c.ok]

    if args.json:
        print(json.dumps({"ok": not failed, "source": "git" if from_git else "walk",
                          "checks": [c.to_dict() for c in checks]}, indent=2))
        sys.exit(1 if failed else 0)

    print("What a `git add .` would publish, checked before it happens.")
    if not from_git:
        print("  NOTE: git did not answer, so this walked the tree instead and "
              "cannot read .gitignore. Treat data and generated findings as "
              "approximate.")
    print()
    for c in checks:
        print(f"  [{'ok  ' if c.ok else 'FAIL'}] {c.name:<13} {c.detail}")
        for offender in c.offenders[:12]:
            print(f"           - {offender}")
        if len(c.offenders) > 12:
            print(f"           ... and {len(c.offenders) - 12} more")
    print()
    if failed:
        print(f"{len(failed)} check(s) failed. Do not publish yet.")
    else:
        print("All checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()