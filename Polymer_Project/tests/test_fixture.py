"""The synthetic fixture can neither overwrite the real export nor pass as real.

Two risks, two guarantees, each with its own tests:

1. Collision. Until 2026-08-16 make_dry_run_fixture.py wrote the exact file
   names export_trials.py writes; one dry run with the real CSVs present would
   have replaced a multi-minute export with noise, silently. The fixture now
   writes `dryrun_`-prefixed names only and refuses the real ones.
2. Provenance. A synthetic input is stamped `is_synthetic=True` in its
   metadata; run_report.py and run_real_case.py read the flag and mark the
   output SYNTHETIC FIXTURE everywhere a reader could look, whatever --label
   was passed. Rule 9: no synthetic number can survive into a real report.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

import make_dry_run_fixture as fixture
import run_real_case
import run_report


# ---------------------------------------------------------------------------
# 1) Collision: the fixture never produces a real export name
# ---------------------------------------------------------------------------


def test_fixture_names_are_disjoint_from_the_real_export_names():
    real = set(fixture.REAL_EXPORT_FILES)
    fake = set(fixture.FIXTURE_FILES.values())
    assert not real & fake
    assert all(name.startswith(fixture.PREFIX) for name in fake)
    # The real names are the ones export_trials.py and run_real_case.py use;
    # pin them here so a rename on either side fails this test, not the user.
    assert real == {"trials_m2_with_rut.csv", "trials_m2_ex_rut.csv", "trials_m2_meta.csv"}
    assert real == set(run_real_case.REAL_FILES.values())


def test_write_fixture_only_writes_prefixed_files(tmp_path):
    written = fixture.write_fixture(out_dir=str(tmp_path))
    on_disk = sorted(os.listdir(tmp_path))
    assert on_disk == sorted(fixture.FIXTURE_FILES.values())
    assert set(map(os.path.basename, written.values())) == set(fixture.FIXTURE_FILES.values())
    for real_name in fixture.REAL_EXPORT_FILES:
        assert not (tmp_path / real_name).exists()


def test_writer_refuses_a_real_export_name(tmp_path):
    frame = pd.DataFrame({"a": [1.0]})
    for real_name in fixture.REAL_EXPORT_FILES:
        with pytest.raises(ValueError, match="real export name"):
            fixture._write_csv(frame, str(tmp_path), real_name)
    assert os.listdir(tmp_path) == []


def test_fixture_keeps_the_export_layout_and_is_stamped_synthetic(tmp_path):
    written = fixture.write_fixture(out_dir=str(tmp_path))
    with_rut = pd.read_csv(written["with_rut"], index_col=0)
    ex_rut = pd.read_csv(written["ex_rut"], index_col=0)
    meta = pd.read_csv(written["meta"])
    assert with_rut.shape == (103, 21)
    assert ex_rut.shape == (103, 21)
    assert list(with_rut.columns) == list(meta["trial_id"])
    assert {"trial_id", "trial_kind", "is_frozen_cell", "sharpe_annual_with_rut",
            "sharpe_annual_ex_rut", "is_synthetic"} <= set(meta.columns)
    assert meta["is_synthetic"].all()
    assert int(meta["is_frozen_cell"].sum()) == 1
    assert (meta["trial_kind"] == "grid").sum() == 18
    assert (meta["trial_kind"] == "gamma_sweep").sum() == 3


# ---------------------------------------------------------------------------
# 2) Provenance: a synthetic input is stamped, a real one is not
# ---------------------------------------------------------------------------


def test_is_synthetic_reads_the_flag_and_only_the_flag():
    assert run_report.is_synthetic(pd.DataFrame({"is_synthetic": [True, True]}))
    assert run_report.is_synthetic(pd.DataFrame({"is_synthetic": ["True", "False"]}))
    assert not run_report.is_synthetic(pd.DataFrame({"is_synthetic": [False, False]}))
    assert not run_report.is_synthetic(pd.DataFrame({"trial_id": ["t00"]}))
    # run_real_case.py must agree with run_report.py on what "synthetic" means.
    for frame in (pd.DataFrame({"is_synthetic": [True]}), pd.DataFrame({"x": [1]})):
        assert run_real_case.is_synthetic(frame) == run_report.is_synthetic(frame)


def _run_report(monkeypatch, capsys, tmp_path, trials, meta, label):
    code = tmp_path / "engine.py"
    code.write_text("pos = signal.shift(1)\n", encoding="utf-8")
    out = tmp_path / "report.html"
    monkeypatch.setattr(sys, "argv", [
        "run_report.py", "--code", str(code), "--out", str(out),
        "--trials", str(trials), "--meta", str(meta),
        "--sharpe-col", "sharpe_annual_with_rut", "--label", label,
    ])
    run_report.main()
    return out.read_text(encoding="utf-8"), capsys.readouterr().out


def test_report_on_the_fixture_is_stamped_synthetic_everywhere(monkeypatch, capsys, tmp_path):
    written = fixture.write_fixture(out_dir=str(tmp_path / "data"))
    # A misleading label must not be able to hide the stamp.
    page, console = _run_report(monkeypatch, capsys, tmp_path,
                                written["with_rut"], written["meta"], "Universe with RUT")
    tag = run_report.SYNTHETIC_TAG
    assert f"<title>{tag} (dry run): Backtest Integrity Audit</title>" in page
    assert f"{tag}: every number on this page is meaningless" in page   # subtitle
    assert f"<h3>{tag}: Universe with RUT</h3>" in page                   # section label
    assert "is_synthetic=True in the metadata" in page                    # provenance
    assert f"WARNING: the deflation input is a {tag}" in console
    assert f"deflation [{tag}: Universe with RUT]" in console


def test_report_on_a_real_layout_is_not_stamped(monkeypatch, capsys, tmp_path):
    with_rut, _, meta = fixture.build_fixture(seed=1)
    meta = meta.drop(columns=["is_synthetic"])   # exactly what export_trials.py writes
    trials = tmp_path / "trials_m2_with_rut.csv"
    meta_path = tmp_path / "trials_m2_meta.csv"
    with_rut.to_csv(trials, index_label="Date")
    meta.to_csv(meta_path, index=False)
    page, console = _run_report(monkeypatch, capsys, tmp_path, trials, meta_path, "Universe with RUT")
    assert run_report.SYNTHETIC_TAG not in page
    assert run_report.SYNTHETIC_TAG not in console
    assert "<title>Backtest Integrity Audit</title>" in page
    assert "<h3>Universe with RUT</h3>" in page


def test_run_real_case_dry_run_reads_only_prefixed_files():
    real = run_real_case.input_files(dry_run=False)
    dry = run_real_case.input_files(dry_run=True)
    assert set(real.values()) == set(fixture.REAL_EXPORT_FILES)
    assert set(dry.values()) == set(fixture.FIXTURE_FILES.values())
    assert not set(real.values()) & set(dry.values())


def test_run_real_case_without_dry_run_never_opens_the_fixture(monkeypatch, tmp_path, capsys):
    """With only the fixture on disk, the real-case script must stop, not
    silently analyse synthetic numbers under a real heading."""
    fixture.write_fixture(out_dir=str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        run_real_case.main([])
    assert "Missing" in str(excinfo.value.code)
    assert "trials_m2_meta.csv" in str(excinfo.value.code)
    assert "SYNTHETIC" not in capsys.readouterr().out


def test_run_real_case_dry_run_prints_the_banner(monkeypatch, tmp_path, capsys):
    fixture.write_fixture(out_dir=str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    run_real_case.main(["--dry-run"])
    out = capsys.readouterr().out
    assert out.startswith("SYNTHETIC FIXTURE")
    assert "SYNTHETIC FIXTURE, UNIVERSE WITH RUT" in out
    assert "SYNTHETIC FIXTURE, UNIVERSE EX RUT" in out
    assert out.count("SYNTHETIC FIXTURE: the input carries is_synthetic=True") == 2
