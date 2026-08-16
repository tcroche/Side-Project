"""The repository audits itself.

Every source file of the auditor must come out clean under its own rules. This
is a living clean-corpus: any future rule change that starts flagging the
tool's own code fails CI immediately, which is how the six false positives
documented in the DEVLOG stay fixed.

`bench/cases` is deliberately excluded: it will hold seeded-bug scripts whose
whole purpose is to fire.
"""

from __future__ import annotations

import glob
import os

import pytest

from auditor.ast_scan import audit_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = sorted(
    path
    for pattern in (
        "core/*.py",
        "auditor/*.py",
        "report/*.py",
        "bench/*.py",
        "tests/*.py",
        "*.py",
    )
    for path in glob.glob(os.path.join(ROOT, pattern))
)


@pytest.mark.parametrize("path", TARGETS, ids=[os.path.relpath(p, ROOT) for p in TARGETS])
def test_own_source_is_clean(path):
    findings = audit_file(path)
    assert findings == [], (
        f"{os.path.relpath(path, ROOT)} fired {[str(f) for f in findings]} -- "
        f"either the code leaks or a rule just regressed in precision."
    )
