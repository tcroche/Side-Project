"""
make_dry_run_fixture.py -- generate a synthetic stand-in for the M2 export.

Run this BEFORE the real export, so that the plumbing (run_report.py,
run_real_case.py) is known to work end to end before spending time on a
21-configuration backtest or money on API calls:

    python make_dry_run_fixture.py
    python run_report.py --code m2_backtester --out rapport_dry.html ^
        --trials data\\dryrun_trials_m2_with_rut.csv ^
        --meta   data\\dryrun_trials_m2_meta.csv ^
        --sharpe-col sharpe_annual_with_rut --label "Dry run (synthetic)"
    python run_real_case.py --dry-run

Two guarantees, one per risk
----------------------------
1. The fixture CAN NOT overwrite the real export. It writes files with the
   same shapes and column layout as export_trials.py but under DIFFERENT
   names, prefixed `dryrun_`. The real export names are listed in
   REAL_EXPORT_FILES and the writer refuses them outright. (Before this, the
   fixture wrote the exact real names: one `python make_dry_run_fixture.py`
   with the real CSVs present would have silently replaced a multi-minute
   export with noise.)
2. A synthetic input CAN NOT pass as real. Every row of the metadata carries
   `is_synthetic = True`; run_report.py and run_real_case.py read that flag
   and stamp SYNTHETIC FIXTURE on the page title, the subtitle, the section
   label and the console, whatever --label was passed.

The numbers are meaningless; only the plumbing is being tested. Delete the
fixture afterwards (rule 9: no synthetic number survives into a real report):

    del data\\dryrun_*.csv       (Windows)
    rm data/dryrun_*.csv         (macOS / Linux)
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

OUT_DIR = "data"
PREFIX = "dryrun_"
N_DAYS = 103  # the real in-sample length
ANN = np.sqrt(252.0)

#: What export_trials.py writes. The fixture must never produce these names.
REAL_EXPORT_FILES = (
    "trials_m2_with_rut.csv",
    "trials_m2_ex_rut.csv",
    "trials_m2_meta.csv",
)

#: What this script writes: same layout, distinct names.
FIXTURE_FILES = {
    "with_rut": f"{PREFIX}trials_m2_with_rut.csv",
    "ex_rut": f"{PREFIX}trials_m2_ex_rut.csv",
    "meta": f"{PREFIX}trials_m2_meta.csv",
}

assert not set(FIXTURE_FILES.values()) & set(REAL_EXPORT_FILES), (
    "fixture file names collide with the real export names"
)


def build_fixture(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (with_rut, ex_rut, meta) with the exact layout of the M2 export."""
    rng = np.random.default_rng(seed)

    specs = []
    for window, k_entry, stop in itertools.product([90, 120, 150], [2.5, 3.0, 3.5], [0.005, 0.01]):
        specs.append(dict(window=window, k_entry=k_entry, stop_loss=stop,
                          gamma=0.3, trial_kind="grid"))
    for gamma in (0.2, 0.3, 0.5):
        specs.append(dict(window=120, k_entry=3.0, stop_loss=0.005,
                          gamma=gamma, trial_kind="gamma_sweep"))
    for i, spec in enumerate(specs):
        spec["trial_id"] = f"t{i:02d}"
        spec["is_frozen_cell"] = (
            spec["window"] == 120 and spec["k_entry"] == 3.0
            and spec["stop_loss"] == 0.005 and spec["trial_kind"] == "grid"
        )

    ids = [s["trial_id"] for s in specs]
    dates = pd.bdate_range("2025-01-06", periods=N_DAYS)

    # Grid trials on the same data are strongly correlated: one common factor
    # plus a small idiosyncratic component. This mirrors reality closely enough
    # for a plumbing test.
    common = rng.normal(0.0, 0.0015, size=(N_DAYS, 1))

    def build(scale: float) -> pd.DataFrame:
        idio = rng.normal(0.0, 0.0006, size=(N_DAYS, len(ids)))
        return pd.DataFrame(common * scale + idio, index=dates, columns=ids)

    with_rut, ex_rut = build(1.0), build(0.7)

    meta = pd.DataFrame(specs)
    meta["sharpe_annual_with_rut"] = [
        with_rut[t].mean() / with_rut[t].std(ddof=1) * ANN for t in ids
    ]
    meta["sharpe_annual_ex_rut"] = [
        ex_rut[t].mean() / ex_rut[t].std(ddof=1) * ANN for t in ids
    ]
    meta["net_return_pct_with_rut"] = [100.0 * with_rut[t].sum() for t in ids]
    meta["is_synthetic"] = True  # the flag the consumers stamp the output with
    meta = meta[[
        "trial_id", "trial_kind", "is_frozen_cell", "window", "k_entry",
        "stop_loss", "gamma", "sharpe_annual_with_rut", "sharpe_annual_ex_rut",
        "net_return_pct_with_rut", "is_synthetic",
    ]]
    return with_rut, ex_rut, meta


def _write_csv(frame: pd.DataFrame, out_dir: str, name: str, **kwargs) -> str:
    """Write one CSV, refusing any of the real export names (defense in depth:
    the constants above already differ, this makes a future rename fail loudly
    instead of silently clobbering the export)."""
    if os.path.basename(name) in REAL_EXPORT_FILES:
        raise ValueError(
            f"refusing to write {name!r}: that is a real export name; the fixture "
            f"must only ever write {PREFIX}* files"
        )
    path = os.path.join(out_dir, name)
    frame.to_csv(path, **kwargs)
    return path


def write_fixture(out_dir: str = OUT_DIR, seed: int = 0) -> dict[str, str]:
    """Build and write the fixture; return {kind: path} for the three files."""
    os.makedirs(out_dir, exist_ok=True)
    with_rut, ex_rut, meta = build_fixture(seed)
    return {
        "with_rut": _write_csv(with_rut, out_dir, FIXTURE_FILES["with_rut"], index_label="Date"),
        "ex_rut": _write_csv(ex_rut, out_dir, FIXTURE_FILES["ex_rut"], index_label="Date"),
        "meta": _write_csv(meta, out_dir, FIXTURE_FILES["meta"], index=False),
    }


def main() -> None:
    paths = write_fixture()
    with_rut = pd.read_csv(paths["with_rut"], index_col=0)
    ex_rut = pd.read_csv(paths["ex_rut"], index_col=0)
    meta = pd.read_csv(paths["meta"])
    print(f"SYNTHETIC FIXTURE written to {OUT_DIR}/ -- the numbers are meaningless.")
    print(f"  {FIXTURE_FILES['with_rut']}  {with_rut.shape}")
    print(f"  {FIXTURE_FILES['ex_rut']}    {ex_rut.shape}")
    print(f"  {FIXTURE_FILES['meta']}      {len(meta)} trials, is_synthetic=True on every row")
    print(f"  (the real export names {REAL_EXPORT_FILES} were not touched)")
    print()
    print("Now run one of:")
    print(f"  python run_report.py --code m2_backtester --out rapport_dry.html "
          f"--trials {paths['with_rut']} --meta {paths['meta']} "
          f"--sharpe-col sharpe_annual_with_rut --label \"Dry run (synthetic)\"")
    print("  python run_real_case.py --dry-run")
    print(f"Then delete the fixture:  del data\\{PREFIX}*.csv   (rule 9)")


if __name__ == "__main__":
    main()
