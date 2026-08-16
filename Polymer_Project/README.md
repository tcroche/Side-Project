# Backtest Integrity Auditor (BIA)

A backtest can be wrong in two independent ways.

**Statistically**, because many configurations were tried and the best one was reported: the
in-sample Sharpe ratio is then inflated by selection, a bias Bailey and López de Prado
quantified and that almost no pipeline computes.

**In the code**, because a leak lets the future into the signal: a negative shift, a centred
rolling window, a scaler fitted before the split, a fill that walks backwards through time.

BIA answers both questions, keeps the two answers apart, and states in writing what it
cannot see. One command produces one self-contained HTML report that opens offline and
prints to A4 PDF from any browser.

> Built for the Polymer Capital Tech Expo 2026 by Théo Crochemar, MSc. Applied Mathematics
> and Quantitative Finance (MMMEF), Université Paris 1 Panthéon-Sorbonne.

---

## The result it was built to produce

The demonstration case is my ownMSc. backtester: an intraday momentum strategy on five
equity indices, in-sample Sharpe **1.92** over 18 tuned configurations. My original write-up
said the Sharpe "collapsed to a deflated 0.92", confusing a probability with a Sharpe ratio.
BIA exists to catch exactly that.

| Question | Answer |
|:--|:--|
| Does the Sharpe survive selection? | **DSR 0.929** against a 0.95 threshold. Rejected. |
| Is the winner overfitted? | **PBO 0.914**, the 99th percentile of its own simulated null. |
| Where did the profit come from? | **69.7 %** of the total P&L is one session, 2025-04-09. **18 of 18** grid cells depend on it. |
| Is the code leaking? | **0 deterministic findings** over 16 files. |
| Anything a rule cannot express? | **2 semantic questions**, both about execution timing, neither established. |

The conclusion that matters: **the code was clean and the leak was in the selection.** Which
is precisely why an auditor needs both halves, and why they must never be merged into one
score.

Supporting numbers from the same run: MinTRL **129** observations with the observed moments
against **301** if returns were normal, so the right tail is buying 172 observations of track
record; skewness **+6.28**, kurtosis **52.6**, variance term **0.429**, meaning PSR is
inflated by a factor of about 1.53; mean pairwise correlation between trials **+0.915**, so
18 trials are nothing like 18 independent bets. Excluding the single asset that carried the
result (RUT), the same pipeline gives DSR **0.378** and PBO **0.357** at the 27th percentile
of its null: indistinguishable from noise, which is not the same thing as clean.

---

## Install and run

Python 3.12.

```bash
git clone <repo-url>
cd Polymer
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m pytest                                     # 267 tests
```

Code audit only, offline, free, no account:

```bash
python run_audit.py path/to/strategy.py
```

The full report, both axes, one file:

```bash
python run_report.py --code my_backtester --out report.html ^
  --trials data\trials.csv --meta data\trials_meta.csv ^
  --sharpe-col sharpe_annual --label "My universe"
```

Add `--llm` to run the semantic pass. It needs `ANTHROPIC_API_KEY` in a gitignored `.env`
(see `.env.example`); responses are cached on disk, so a second identical run costs nothing
and replays the same raw model output. Without a key, or with `AUDITOR_OFFLINE=1`, the
semantic section renders as **Not run** with its reason, and the rest of the report is
unaffected.

**Input format.** `--trials` is a CSV of daily returns, one column per configuration, dates
in the first column. `--meta` is one row per configuration with at least `trial_id`,
`trial_kind` and `is_frozen_cell`, plus the column named by `--sharpe-col`. `export_trials.py`
shows how to produce both from a backtester.

---

## What the report contains

Five sections that are never merged, each stamped with where its claims come from:

| Section | Provenance | Line quality |
|:--|:--|:--|
| 01 Statistical deflation | arithmetic in this repository | solid, ticked |
| 02 Deterministic code findings | Python AST, exact line numbers | solid, ticked |
| 03 Semantic findings, to verify | a language model, unverified | **dashed, no tick** |
| 04 What this audit cannot see | outside any static analysis | no rule, no tick |
| 05 Provenance | the command, inputs and versions that produced the page | |

The tiers are separated by line quality before colour, because line quality survives
grayscale printing and colour blindness. The tick is the auditor's "vouched to source" mark
and is deliberately absent from the model tier: the tool ticks what it verified and nothing
else. Every mark on both charts plots a number the tool computed, and every charted figure
also appears as text beside it.

A section that did not run says **Not run**, with the reason, plus the sentence that an
absence is not a clean bill of health. An empty section that reads as "all good" is the main
failure mode of an audit tool.

---

## Axis 1: statistical deflation

Implemented from Bailey and López de Prado (2012, 2014) and Bailey, Borwein, López de Prado
and Zhu (2017):

- **PSR**, the probability the true Sharpe ratio exceeds a benchmark, corrected for skewness,
  kurtosis and sample length;
- **E[max SR]** under the null that no trial has skill, so the winner is compared against what
  luck alone produces over N attempts;
- **DSR**, the same probability deflated by that expectation, reported at several N and with
  two estimators of V[SR] (empirical across trials, and marginal 1/T), because the choice
  changes the answer and hiding it would be dishonest;
- **MinTRL**, the track record needed to clear the threshold, with and without the
  non-normality of the actual returns;
- **CSCV / PBO**, the probability the in-sample winner ranks below median out-of-sample,
  positioned inside **its own simulated null** at the user's exact dimensions. A PBO near 0.5
  does not mean clean: it can mean there was nothing to overfit;
- **concentration diagnostics**: share of the P&L in the best observation, Sharpe with it
  removed, how many observations take the total to zero, and how many configurations depend
  on the same date.

PSR, DSR and PBO are **probabilities in [0, 1], never Sharpe ratios**. A `UnitError` is
raised, not warned, when an annualized Sharpe arrives where a per-period one is expected.

---

## Axis 2: code leakage

**Eight deterministic AST rules**, offline, free, exact line numbers, same answer every run:

| Rule | What it matches |
|:--|:--|
| R1 | Negative shift on a feature |
| R2 | Centred rolling window |
| R3 | Preprocessor fitted before the train/test split |
| R4 | Possible same-bar execution |
| R5 | Feature built from the target |
| R8 | Backward fill or interpolation over time |
| R9 | Model or search fitted on test data |
| R10 | Normalisation by whole-sample statistics |

R6 and R7 (survivorship, non-point-in-time data) were **removed**. They are properties of the
data, not of the code: the code that reads a good ticker list is identical to the code that
reads a bad one. They live in the manual checklist of section 04, phrased as questions, and a
test forbids claiming them as rules.

**One semantic pass by a language model**, for what a syntax rule cannot express: a
`merge_asof` whose direction lets rows see later events, a label whose definition quietly
reads past the decision time, a custom function that looks ahead without using any banned
keyword.

Nothing the model says is taken on faith:

- line numbers are checked against the real file; a finding citing a line that does not exist
  is rejected and counted;
- the snippet shown is extracted from the real source by the tool, never copied from the
  model's output;
- **high and medium must be earned.** A finding is only eligible when the model explicitly
  declares `external_dependency: null`. A declared dependency, or a missing field, caps the
  severity at review deterministically, in `ground_findings()`, with the original claim
  recorded. Omission never buys severity;
- prompts are versioned in a YAML registry with a changelog, never edited in place after a
  benchmark has run against them;
- rejection and capping rates are printed and written to the report, so the disagreement rate
  between model and harness is measurable rather than assumed.

That cap used to be a prompt instruction. It was violated on first contact with real code, in
three ways at once, with a suggested fix that would have degraded a correct causal engine.
**A prompt rule is a request; code is a guarantee.** The constraint moved into the harness and
the prompt was bumped to v1.2.0.

---

## Benchmark

16 seeded-bug cases: 8 trapped files carrying 9 catalogue leaks, 2 semantic leaks no
syntactic rule can express, 5 clean controls and 1 file whose correct answer is a question.
Ground truth lives in `bench/truth.py` and is re-verified by the test suite, so the printed
numbers cannot drift from the detectors silently.

Measured 2026-08-12, `python run_bench.py --llm`:

| Detector | Leaks found (/11) | Catalogue (/9) | Semantic (/2) | False positives on 6 controls |
|:--|--:|--:|--:|--:|
| AST only | 9 | 9 | 0 | 0 |
| LLM only (v1.2.0) | 9 | 7 | 2 | 0 |
| **Hybrid (union)** | **11** | 9 | 2 | **0** |

**How to read this without overclaiming.** The honest headline is 11 of 11 seeded leaks, 0
false positives on 6 controls, and above all the **complementarity**: the AST layer is perfect
on its own syntactic territory and blind outside it, the model is the reverse. The hybrid's 18
true positives are *agreement*, not 18 discoveries: 7 leaks are found by both halves and
counted twice by the stated convention.

Two measurements that matter more than the table:

- **Exclusion compliance: 2 of 8.** The model re-reported 6 of the 8 patterns the prompt
  explicitly told it to leave to the rules. So its "7 of 9 catalogue" recall **is not
  interpretable**: obedience and miss cannot be distinguished. This is the central argument
  for the hybrid architecture, and the reason the deterministic layer is the floor on its own
  territory.
- **The benchmark audits its author.** Seeding a trap revealed a precision gap in R1, which
  fired on label construction, the one negative shift its own fix text calls legitimate. If a
  trapped case contains anything other than its seeded leak, the measured precision describes
  the author, not the tool.

And a caveat stated once: the benchmark is small and in-distribution by construction. We wrote
the detectors, the prompt and the cases. These numbers measure the coherence of the design,
not field performance. The LLM rows are reproducible **from the cache**, since post-processing
is deterministic; temperature 0 does not make a hosted model bit-reproducible across fresh
calls. `run_bench.py --llm --no-cache` would measure run-to-run variance. Not done.

Observed once on real code, outside the benchmark: three files with an identical loop shape,
one flagged. The semantic pass is not consistent across structurally identical inputs, which
is the same lesson from the other side.

---

## What it cannot see

Printed in every report, as questions, never as detections:

1. Is the trading universe defined from constituents as of each date, or from today's index
   membership?
2. Are fundamental or accounting inputs point-in-time, or restated values attributed to their
   original date?
3. Do corporate actions, delistings and halted names appear in the history, or were they
   dropped when the data was pulled?
4. Does the transaction-cost model reflect the liquidity actually available at the size being
   simulated?
5. How many configurations were tried in total, including the ones that were abandoned and
   never written down?

Plus one structural limitation: the semantic pass reads **one file at a time**. A leak that
only exists in the interaction between a signal module and its engine can be raised as a
question but cannot be established. Multi-file context is future work, and saying so is part
of the tool.

---

## Design rules, each paid for by a mistake

1. A prompt rule is a request; **code is a guarantee**.
2. Grounding is verified by code, never asserted by the model, and the rejection and capping
   rates are exposed.
3. DSR, PSR and PBO are probabilities in [0, 1], never Sharpe ratios. Units are enforced by an
   exception, not a comment.
4. The two detectors are never merged. Overlap is **annotated** on the semantic side
   ("corroborates R1 at line 11") and counted; the deterministic list is read-only.
5. No number without a reproducible script. Every figure in the report is measured during the
   run, never copied from a previous one.
6. Negative instructions are weakly obeyed, measured at 2 of 8. Any discipline that matters
   lives in code.
7. What did not run says so, with its reason.
8. No synthetic number can survive into a real report: the dry-run fixture writes
   `dryrun_`-prefixed files it is impossible to confuse with a real export, stamps
   `is_synthetic` on every row, and the report marks itself **SYNTHETIC FIXTURE** in five
   places whatever label was passed.
9. Nothing the tool writes uses an em dash; text quoted from a model is reproduced character
   for character, because a quotation is evidence.

The repository audits itself: every source file must come out clean under its own rules,
inside the test suite, forever. Six false positives found that way are now a permanent
regression corpus.

---

## Layout

```
core/        units.py, stats.py, cscv.py, concentration.py     the statistical axis
auditor/     ast_scan.py, schema.py, llm_pass.py, cache.py     the code axis
prompts/     code_auditor_v1.yaml                              versioned prompt registry
bench/       cases/, truth.py, score.py                        the seeded-bug benchmark
report/      deflation.py, corroboration.py, render.py         the report layer
docs/        build_writeup.py                                  the one-page PDF write-up
tests/       267 tests
run_audit.py  run_report.py  run_bench.py  run_real_case.py  run_deflation_demo.py
make_dry_run_fixture.py  export_trials.py
```

---

## Data and keys

No API key, no dataset and no generated report is committed. The 1-minute index data behind
the demonstration case is course-provided and is not redistributed; `data/*.csv`,
`data/llm_cache/` and the generated `.html` files are gitignored. `make_dry_run_fixture.py`
produces a synthetic stand-in with the exact shape of a real export so the whole pipeline can
be exercised end to end without any data at all.

---

## References

- Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
  *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*, 40(5), 94-107.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of
  Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39-69.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4),
  36-52.

## Author

**Théo Crochemar**, MSc. Applied Mathematics and Quantitative Finance (MMMEF EFG : track), Université Paris 1
Panthéon-Sorbonne. GitHub: [tcroche](https://github.com/tcroche).
