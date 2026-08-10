# DEVLOG — Backtest Integrity Auditor

Dated engineering journal. Written as I go, not reconstructed afterwards.
This file is the raw material for the "Iterations / Reflections" section of the
submission, and for the disclosure of AI coding tools that Polymer asks for.

Format per entry: what I set out to do, what actually happened, what I decided,
what I would do differently.

---

## 2026-08-10 — Day 0: scaffold and statistical core

**Goal.** Stand up the repository and get the deflation mathematics right and
tested before writing anything else. Nothing about this project is credible if
the statistics are wrong.

**AI coding tools used.** Claude (Anthropic) via the chat interface, for the
scaffold, the first implementation of `core/stats.py` and `core/units.py`, and
the test suite. I reviewed every formula against the source papers before
accepting it. Prompts and iterations recorded below.

**What I built.**
- `core/units.py` — Sharpe frequency conversion plus a guard that raises
  `UnitError` when a value that looks annualized is passed where a per-period
  value is expected.
- `core/stats.py` — PSR, E[max SR], DSR, MinTRL, a heuristic effective-trial
  count under correlation, and a `DeflationReport` that narrates every
  intermediate quantity with its unit.
- `tests/test_stats.py` — 35 tests: analytic cases recomputed independently,
  monotonicity and scaling invariants, and two known-truth end-to-end cases.
- `run_deflation_demo.py` — four scenarios where the right answer is known by
  construction.

**What I got wrong, and fixed.**

1. *DSR is a probability, not a Sharpe ratio.* My earlier project write-up said
   "an in-sample Sharpe of 1.93 collapsed to a deflated 0.92". That mixes units:
   DSR ∈ [0,1] is P(true SR > E[max SR]). My own M2 README had it right
   (DSR = 0.92 against a 0.95 threshold, i.e. a rejection) — the write-up had
   drifted. Everything user-facing now labels PSR and DSR explicitly as
   probabilities, and `DeflationReport.to_text()` prints the words "NOT a Sharpe
   ratio" on those two lines. The bug I most wanted to catch was in my own prose.

2. *The √(2 ln N) test was wrong as written.* I first asserted that
   E[max SR] / √(2 ln N) sits within 10% of 1. It does not: convergence is slow
   and strictly from below (0.876 at N=1000, 0.915 at N=100 000), because the
   second-order term is negative. Replaced with two better tests: monotone
   convergence from below, and a direct comparison against the exact
   E[max of N standard normals] obtained by numerical integration of
   ∫ x·N·φ(x)·Φ(x)^(N−1) dx. Relative error of the Bailey–López de Prado
   approximation is 1.5% at N = 30 and 0.4% at N = 1000. Now I can state the
   accuracy of my own implementation rather than assert it.

**Design decision — why the unit guard raises rather than warns.** A per-period
Sharpe above 1.0 implies an annualized Sharpe between 3.5 (monthly) and 15.9
(daily). It is almost always a mistake. `strict_units=False` exists as an escape
hatch, and the demo uses it to show what the mistake produces: PSR = 1.000000,
a manufactured certainty.

**Design decision — the demo must contain a PASS.** A tool that only ever
rejects proves nothing. Scenarios B and C use the same true edge with 500 and
1000 days of history: reject, then accept. MinTRL quantifies the gap
(665 days needed at T = 500). This makes the rejection of my own M2 backtest
meaningful rather than merely conservative.

**Open question for tomorrow.** Reproducing DSR = 0.92 on the real M2 case
requires the 30 in-sample trial Sharpe ratios from `calibrate_is.py`, which
were never persisted. Task: export them to `data/trials_m2.csv`.

**Time.** ~4 h.

---

## 2026-08-11 — Day 1: CSCV / PBO and the real case

*(to fill in)*

---
