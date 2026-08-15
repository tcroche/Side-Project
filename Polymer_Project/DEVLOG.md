# DEVLOG - Backtest Integrity Auditor

Dated engineering journal. Written as I go, not reconstructed afterwards.
This file is the raw material for the "Iterations / Reflections" section of the
submission, and for the disclosure of AI coding tools that Polymer asks for.

Format per entry: what I set out to do, what actually happened, what I decided,
what I would do differently.

---

## 2026-08-10 - Day 0: scaffold and statistical core

**Goal.** Stand up the repository and get the deflation mathematics right and
tested before writing anything else. Nothing about this project is credible if
the statistics are wrong.

**AI coding tools used.** Claude (Anthropic) via the chat interface, for the
scaffold, the first implementation of `core/stats.py` and `core/units.py`, and
the test suite. I reviewed every formula against the source papers before
accepting it. Prompts and iterations recorded below.

**What I built.**
- `core/units.py` : Sharpe frequency conversion plus a guard that raises
  `UnitError` when a value that looks annualized is passed where a per-period
  value is expected.
- `core/stats.py` : PSR, E[max SR], DSR, MinTRL, a heuristic effective-trial
  count under correlation, and a `DeflationReport` that narrates every
  intermediate quantity with its unit.
- `tests/test_stats.py` : 35 tests: analytic cases recomputed independently,
  monotonicity and scaling invariants, and two known-truth end-to-end cases.
- `run_deflation_demo.py` : four scenarios where the right answer is known by
  construction.

**What I got wrong, and fixed.**

1. *DSR is a probability, not a Sharpe ratio.* My earlier project write-up said
   "an in-sample Sharpe of 1.93 collapsed to a deflated 0.92". That mixes units:
   DSR ∈ [0,1] is P(true SR > E[max SR]). My own M2 README had it right
   (DSR = 0.92 against a 0.95 threshold, i.e. a rejection), the write-up had
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

**Design decision - why the unit guard raises rather than warns.** A per-period
Sharpe above 1.0 implies an annualized Sharpe between 3.5 (monthly) and 15.9
(daily). It is almost always a mistake. `strict_units=False` exists as an escape
hatch, and the demo uses it to show what the mistake produces: PSR = 1.000000,
a manufactured certainty.

**Design decision - the demo must contain a PASS.** A tool that only ever
rejects proves nothing. Scenarios B and C use the same true edge with 500 and
1000 days of history: reject, then accept. MinTRL quantifies the gap
(665 days needed at T = 500). This makes the rejection of my own M2 backtest
meaningful rather than merely conservative.

**Open question for tomorrow.** Reproducing DSR = 0.92 on the real M2 case
requires the 30 in-sample trial Sharpe ratios from `calibrate_is.py`, which
were never persisted. Task: export them to `data/trials_m2.csv`.

**Time.** ~4 h.

---

## 2026-08-10 - Day 0.5: CSCV / PBO and the real case

**Goal.** Implement PBO, and build the bridge between the M2 repository and this
one so the real case can be deflated tomorrow morning without surprises.

**Reading `calibrate_is.py` properly.** Three things I had wrong or vague:

1. *The grid is 18 cells, not 30.* `itertools.product(WINDOWS, K_ENTRYS, STOPS)`
   = 3 x 3 x 2 = 18, plus a 3-point gamma sweep = 21. My write-up said "~30
   configurations". `dsr.py` already reports DSR at N in {18, 21, 30}, which is
   the right instinct, N is genuinely uncertain because exploration happened in
   `diagnostics.py`, `compare_trend.py` and `compare_strategies.py` as well. The
   honest statement is "18 recorded in the calibration grid, roughly 30 across
   the whole project", and the tool reports DSR as a function of N rather than a
   single number.

2. *`eval_cell` returns only scalar Sharpe ratios.* Fine for DSR, useless for
   CSCV, which needs the T x N matrix of daily returns. Hence `export_trials.py`.

3. *The gamma sweep is not really three extra trials.* `calibrate_is.py` itself
   shows the Sharpe ratio is invariant to gamma, because tanh sizing frozen at
   entry is a pure leverage factor. Configurations that cannot change the
   ranking add almost no selection room. Counting them is conservative, not
   accurate, the export tags them with `trial_kind` so both counts can be run.

**Estimator parity.** `dsr.py` uses the plain moment ratios `mean(z**3)` and
`mean(z**4)` with `z` standardized by the ddof=1 standard deviation. `scipy`
defaults to the bias-corrected estimators, which differ by about
`sqrt(n(n-1))/(n-2)` = 1.5% on skewness at n = 103. Added a `moment_estimator`
argument so the original figure is REPRODUCED rather than approximated, and set
`run_real_case.py` to `"simple"`. Neither estimator is wrong; publishing a
number without saying which one produced it is.

**Two estimators of V[SR].** E[max SR] needs the variance of trial Sharpe
ratios. The usual estimate is the empirical variance across trials, but that
conflates sampling noise with genuine differences between configurations, while
the derivation assumes every trial has a true Sharpe of zero. Added
`sr_variance_under_null(T)` = (1 + SR^2/2)/T, which is 1/T = 0.0097 at T = 103.
Reporting both, plus their ratio, turns "how did you estimate V[SR]?" from an
awkward question into a line of output.

**The finding that changes how I will present PBO.** My first known-truth test
gave PBO = 0.73 on pure noise and I assumed a bug. Averaged over 12 datasets it
is 0.50, exactly as theory says. The 0.73 was one draw. So I simulated the null
at my own dimensions (T = 103, N = 18, S = 8): mean 0.49, **standard deviation
0.21**, 90% interval roughly [0.16, 0.87]. A single observed PBO at this sample
length is nearly uninformative. Rather than drop the module, `cscv.py` now ships
`pbo_null_distribution()` and `pbo_percentile()` so any observed PBO is reported
as a percentile of its own null. Reporting the null distribution of my own
diagnostic is a better answer than reporting the diagnostic.

**Second finding: CSCV has limited power.** A configuration with a per-period
Sharpe of 0.15 among 19 noise competitors gives mean PBO = 0.35, not 0. It is
only about two standard errors above the best of the competitors. The
known-truth test now uses a deliberately large edge, and a second test documents
where power collapses. Better to state the limitation than to pick a lucky seed.

**Block length.** `suggest_n_blocks(103)` returns 8, giving 12 observations per
block and C(8,4) = 70 combinations. The paper's S = 16 would give 6 rows per
block, which is below the point where a within-block Sharpe means anything,
`cscv_pbo` refuses by default and explains why.

**Built.** `core/cscv.py`, `tests/test_cscv.py` (20 tests), `export_trials.py`
for the M2 repo, `run_real_case.py`, `make_dry_run_fixture.py`. Suite is at
55 tests, all green. The whole real-case pipeline was dry-run on a synthetic
fixture with the exact shape of the real export, and one bug was caught that way
(the metadata loader was consuming `trial_id` as an index).

**Time.** ~3h

---
## 2026-08-11 - Day 1: AST rules
**The M2 figures are reproduced.** DSR = 0.9290 at N = 18 with RUT (the README
said 0.92) and 0.3784 at N = 30 ex RUT (the README said 0.38). Both original
numbers come back, so the refactored core is faithful to `dsr.py`.
 
**Finding 1 : the result is one day.** The frozen cell's in-sample P&L is
+2.572% in total, and **2025-04-09 alone contributes +1.792%, i.e. 70%**.
Only 39.8% of days are profitable and the median day is negative. Removing the
best three days takes the total below zero. The Sharpe ratio falls from 1.92 to
1.11 without that single session. Ex RUT it is worse: removing the best day
takes the Sharpe from +0.90 to **−0.47**.
 
And the part that settles it: **18 of the 18 grid cells draw more than half
their in-sample P&L from that same day.** The grid was never exploring
eighteen strategies. It was re-expressing one event eighteen times. That is
also why the mean pairwise correlation between trials is 0.915.
 
This is not something the DSR tells you. It needed a new module,
`core/concentration.py`, and it is now the most legible output the tool
produces, no statistics background required to understand "70% of the profit
came from one day".
 
**Finding 2 : the non-normality correction is HELPING the strategy.** Skewness
is +6.28 and kurtosis 52.6. The PSR variance term
`1 − g3·SR + ((g4−1)/4)·SR²` comes to 0.429, which is below 1, so it *divides*
the z-score by 0.655 and inflates the probability by 1.53x. Positive skew is
rewarded by PSR, which is correct in principle and misleading here, because a
skewness of +6.28 estimated on 103 observations is an artefact of one outlier,
not a distributional property. Concretely: MinTRL against E[max SR] is 129 days
with the observed moments and **301 days if the returns were normal**. The fat
right tail is buying 172 days of apparent track record. Both numbers are now
printed side by side.
 
**Finding 3 : I had a double-counting bug, and the data exposed it.** With
rho_bar = 0.915 my `effective_number_of_trials` heuristic returned N_eff = 1.1,
which wiped out the deflation entirely (DSR back up to 0.9688). That is wrong.
Writing each trial as `e_i = sigma(sqrt(rho)Z + sqrt(1−rho)Y_i)`, the maximum is
`sigma·sqrt(1−rho)·max Y_i`, so correlation acts through the VARIANCE, not
through N. And the cross-sectional sample variance of the N estimated Sharpe
ratios already has expectation `sigma²(1−rho)`, because the common component
cancels out of a cross-sectional variance. Verified by simulation at
rho = 0, 0.5, 0.915: the ratio of empirical to marginal variance comes out at
1.015, 0.513, 0.087 against a theoretical 1−rho of 1.0, 0.5, 0.085.
 
So an empirical V[SR] already absorbs the correlation and must NOT be adjusted
again. `deflate` now takes `v_sr_source`, applies the `(1−rho)` shrink only to
a marginal variance, and reports N_eff as a description of how much independent
exploration the grid contains rather than as an input. The two routes now
bracket sensibly: empirical gives DSR = 0.929, the marginal route corrected for
correlation gives 0.852, versus 0.170 uncorrected.
 
**Finding 4 : PBO splits the story cleanly.** With RUT: PBO = 0.914, at the 99th
percentile of its own simulated null, median out-of-sample rank of the winner
0.211, degradation slope −1.22. Ex RUT: PBO = 0.357, 27th percentile, i.e.
indistinguishable from noise. The asymmetry is the interesting part. A PBO near
0.5 does not mean "clean", it can mean "there was nothing to overfit to". With
RUT there was a pattern and the selection latched onto it; without RUT every
configuration is equally poor everywhere, so selection is a coin flip.
 
**AST scanner built.** Eight rules (R1, R2, R3, R4, R5, R8, R9, R10). R6 and R7
from the original catalogue were dropped: survivorship bias from current index
constituents and non-point-in-time fundamentals are properties of the DATA, and
the code that reads a correct ticker list is byte-identical to the code that
reads a wrong one. They now appear in a manual checklist printed as questions,
never as detections, and a test asserts they are not claimed as rules.
 
**Two false positives caught before shipping.**
 
1. `roll = df["close"].rolling(30)` followed by `z = (x − roll.mean()) / roll.std()`
   fired R10, because the division expression contains no `rolling` call, the
   window was bound to a variable one line earlier. The scanner now tracks names
   bound to windowed objects.
2. `position = pd.Series(model.predict(X_test), index=y_test.index)` fired R5.
   Aligning on `y_test.index` reads the target's labels, not its values. Names
   whose path contains a metadata attribute (`index`, `shape`, `columns`, ...)
   are now excluded. This cut the trapped-script findings from 11 to 9, all
   nine correct.
**Audit of my own M2 engine: zero findings.** Positions are decided at `t` and
applied to `P[t+1] − P[t]`, the rolling windows are trailing, and
`inverse_vol_weights` fits on the in-sample slice and freezes. That is the demo:
*the code was clean; the leak was in the selection and in one April session.
Which is exactly why the tool needs both halves.*
 
**Time.** ~5 h. Suite at 109 tests.
 
---
## 2026-08-11 - Day 1: LLM semantic pass
 
**Goal.** The semantic layer: everything the AST rules cannot express, with the
grounding guarantee enforced by code rather than asserted by the model.
 
**Built.**
- `prompts/code_auditor_v1.yaml` : versioned prompt registry entry (version,
  date, model, temperature 0, changelog, worked examples), in the format the
  Polymer brief asks to see. The system prompt EXCLUDES the AST rules'
  territory by name (shift, center=True, bfill, fit-on-test, whole-sample
  z-scores, position*return), so the model only hunts what the rules cannot:
  `merge_asof` direction errors, forward-looking label construction, custom
  functions that read ahead without banned keywords.
- `auditor/llm_pass.py` : client abstraction (real Anthropic client built only
  when a key exists; a stub in tests), strict JSON parsing (markdown fences
  tolerated, prose rejected wholesale rather than repaired), and
  `ground_findings`, where the guarantees live.
- `tests/test_llm_pass.py` : 21 tests, none touching the network.
**The trust model, concretely.** Line numbers are checked against the real
file; invented lines are rejected and counted. The snippet shown next to each
finding is extracted from the actual source by my code, the model's claimed
snippet is never displayed. Unknown severities coerce to "review". A client
exception degrades to AST-only instead of crashing. Rejections are kept in the
JSON output so the rejection RATE is measurable, that number goes in the
benchmark table.
 
**Definition of done, made executable.** The B3 criterion was "0 findings with
a nonexistent line number over 20 runs". That is now literally a test:
`test_definition_of_done_no_invented_line_survives_twenty_runs` feeds 20
scripted responses, each containing an invented line, and asserts none survives
grounding. It runs in CI forever, not once on a good day.
 
**Offline mode.** `AUDITOR_OFFLINE=1` or a missing key skips the pass with an
explanatory note and leaves the deterministic findings untouched. The demo has
a network-failure story built in.
 
**Design decision : separate sections, stated epistemology.** LLM findings
print under "SEMANTIC FINDINGS (LLM) - TO VERIFY" with the sentence "each one
is a question, not a verdict". Deterministic and probabilistic findings are
never mixed, which is also the honest answer to "how do you deal with
hallucinated findings?".
 
**Time.** ~2.5 h. Suite at 130 tests.

## 2026-08-11 : first contact with real code, on both sides
 
**The semantic pass met the real M2 repository.** 17 files, 3 findings
accepted, **0 rejected by grounding**, the line-verification machinery has now
held on real model output, not just on scripted stubs. Analysis of the three:
 
1. `compare_trend.py` [review] : "does `load_ticker_series()` apply
   whole-sample transformations before the IS filter?" Resolved by reading
   `data_loader.py`: the only fill is `ffill()`, which is causal. Verdict:
   clean, and the model was right to ask rather than assert.
2. `dsr.py` [review] : "does `eval_cell()` enforce an IS-only window?"
   Resolved by reading `calibrate_is.py`: `run_backtest(..., end=IS_END)` is
   hardcoded inside `eval_cell`. Clean. Same pattern: a per-file analyzer
   flagged a cross-file dependency as a question.
3. `strategy_trend.py` [HIGH] : same-bar execution claim. This one is a
   **severity miscalibration**. The model's own text says the leak exists
   "when the caller multiplies pos[t] by the return of bar t", i.e. it is
   conditional on a caller it could not see. The actual caller,
   `backtester.py`, computes `pos[:-1] * np.diff(P)`: the position decided at
   t earns P[t+1]−P[t]. Causal. By the prompt's own rule 3 this should have
   been "review". The tool's design already contained the mitigation, the
   finding sat in the "TO VERIFY" section, never merged with deterministic
   results, but the prompt needed to encode the lesson.
**Prompt bumped to 1.1.0**, driven by that miss: new rule 6 (a finding whose
leak depends on code outside the file is capped at "review" and must name the
exact external fact to check) and a worked example of a loop engine with an
explicit held-over-[t,t+1] convention that must yield no findings. The
registry did what a registry is for: the old prompt text stays in git and in
the changelog, and tomorrow's benchmark will measure 1.1.0, not a moving
target. Lesson recorded: **per-file analysis is the semantic pass's structural
limitation**; multi-file context goes in "future enhancements", honestly.
 
**Then the tables turned: I audited the auditor.** `run_audit.py .` over this
repository produced **6 findings, all false positives**, in exactly two
classes:
 
- *Target-vocabulary homonyms used non-numerically.* `targets` (AST assignment
  targets in `ast_scan.py` itself), `labels` and `label` (axis labels in
  `concentration.py` and `run_real_case.py`) fired R5, though none of them
  moves target VALUES into a feature, they are iterated, joined, or passed to
  `getattr`.
- *R10 mistaking a Sharpe ratio for a normalisation.* `matrix.mean(axis=0) /
  matrix.std(axis=0)` is a cross-sectional Sharpe, a legitimate whole-sample
  computation, but any mean/std division matched the old heuristic.
**Fixes, both principled rather than special-cased.** R5 now classifies HOW a
target reference is used, via a parent-map walk-up: arithmetic, comparisons,
method chains, subscripts and plain aliasing count as value uses; comprehension
iterables, `str`/`len`/`getattr`-style introspection and metadata attributes
(`.index`, `.shape`) do not. R10 now requires the z-score SHAPE, a
subtraction in the numerator, so re-centring fires and ratios of statistics
do not. One regression during the rework (`y_test.index` briefly counted as a
value use through the Attribute branch) was caught by the existing alignment
tests before it ever shipped; the fix routes metadata attributes to non-value.
 
**The six false positives are now a permanent regression corpus**
(`TestSelfAuditFalsePositives`, each snippet lifted verbatim from the file
that fired), and `tests/test_self_audit.py` audits every source file of the
repository inside CI, the tool must stay clean under its own rules forever,
parameterized per file so a regression names the exact file. Trapped-script
check after the tightening: still 9/9 true positives.
 
**Quality note for the benchmark.** Today produced real measured numbers to
report alongside tomorrow's synthetic ones: grounding rejection rate 0/3 on
real code, AST false-positive rate 6→0 on a 30-file adversarial clean corpus
(this repo), true-positive retention 9/9 on the trapped script.
 
**Time.** ~2 h. Suite at 162 tests.
 
---
## 2026-08-12 - Day 2: the cap moves from the prompt into the code
 
**Re-ran the semantic pass on `strategy_trend.py` under prompt v1.1.0, the
version written specifically to fix yesterday's miscalibrated finding, and it
violated its own rule 6 three ways at once.** The finding came back "medium"
instead of "review"; it named no external fact to check; and it was phrased as
a verdict ("This *is* a look-ahead bias") one sentence after hedging its own
premise ("pos[t] is then *implicitly* used..."). Worse, the suggested fix was
wrong for the actual code: `backtester.py` computes `pos[:-1] * np.diff(P)`,
so `pos[t]` already earns `P[t+1] − P[t]`; shifting the position by one bar,
as the model prescribed, would delay a causal signal and degrade a correct
engine. There is no leak here, the information set at the decision is
{P₀…P_t} and the realized return is P_{t+1} − P_t. What the model touched is
a legitimate *execution-assumption* question (can one really fill at P_t
after observing P_t?), which is cross-file by nature and worth exactly
"review".
 
**The lesson, stated once and built on: a prompt rule is a request; code is a
guarantee.** Design principle 5 already said grounding is verified by the
code, never asserted by the model, but severity *entitlement* was still
being asserted, not verified. Fixed structurally:
 
- The JSON schema (prompt **v1.2.0**) gains a required `external_dependency`
  field. `null` means "the leak is established within this file alone".
- `ground_findings()` now enforces the entitlement deterministically:
  **high/medium must be earned** by an explicit `external_dependency: null`.
  A declared dependency, *or a missing field*, caps the severity at
  "review", records the original claim in `capped_from`, and increments a
  `capped` counter exposed in the JSON next to the rejection counter, so the
  disagreement rate between model and harness is measurable. Omission is
  never a path to a higher severity.
- Vocabulary coercion and the cap stay distinct: an out-of-vocabulary
  severity is a vocabulary problem, not an entitlement problem, and does not
  inflate the counter. A model that already said "review" agreed with the
  policy and is not counted either.
- Prompt v1.2.0 also adds rule 7 (fixes for dependent findings must be
  conditional on the named external fact, the blind fix above is the worked
  counter-example) and a fourth example distilled from the real miss: a
  signal-only loop with no visible engine, whose expected answer is a
  "review" naming the consumer's P&L convention.
**Caching, built tonight because the benchmark needs it tomorrow.**
`auditor/cache.py` wraps any client and memoizes `complete()` on disk, keyed
on the sha256 of the *exact* (system, user) pair. By construction a prompt
bump or a one-character source change can never be served a stale response;
identical re-runs never re-bill. Corrupt entries degrade to a miss and heal.
Hit/miss counters are printed after `run_audit.py --llm`, same philosophy as
the rejection rate: measured, not assumed. This is also the ablation
infrastructure: one set of API calls, two post-processings
(`enforce_external_cap=True/False`), so tomorrow's benchmark can report
"N severities corrected, 0 detections lost" from identical model outputs,
the cap changes calibration, never localization.
 
**Verification.** Replayed this morning's real payload (medium, no field,
verdict) through the new pipeline: capped to review with `capped_from:
medium`; ablation mode reproduces the v1.1.0 behaviour exactly. The two new
source files enter the self-audit corpus automatically (23 files now). Suite:
**187 tests**, all green (was 162; +15 llm_pass, +8 cache, +2 self-audit).
 
**Time.** ~1 h.
 
## 2026-08-12 - Day 2 build: the seeded-bug benchmark
 
**16 cases in `bench/cases/`, one category more than the spec.** 8 trapped (9
seeded catalogue leaks, R1, R2, R3, R5, R8, R9, R10, plus one double), 5
clean controls, 2 semantic leaks no syntactic rule can express, and 1
**dependent** case, a category that did not exist when the benchmark was
specified, created by the strategy_trend episode: a signal-only file whose
causality hinges on an unseen caller, where the correct answer is a *question*
at "review" naming the external convention. Not a verdict, and not silence.
The two semantic seeds are chosen to sit exactly in the AST's blind spots:
`groupby(day).transform("mean")` (a whole-bucket aggregate that R10's window
whitelist treats as trailing) and a loop that reads `prices[t + horizon]`
with no banned keyword anywhere.
 
**Building the benchmark caught two real defects before measuring anything.**
 
1. *R1 had a precision gap.* Seeding trap07 (`target =
   close.pct_change().shift(-1)` then `edge = target.rolling(5).mean()`)
   made R1 flag the label construction, the one negative shift its own fix
   text calls legitimate ("only legitimate when building the TARGET"). The
   documentation prescribed an exemption the code never implemented. Fixed:
   a negative shift inside a *bare* target-named assignment no longer fires;
   re-use of that label as a feature remains R5's territory (and R5 does
   fire on trap07's line 12). The exemption is deliberately narrow,
   `df['target'] = ...` still fires, because when unsure a leak detector
   keeps firing. Five new rule tests pin all of this down.
2. *Case-design artifacts are a failure mode of benchmarks themselves.* My
   first trap05 used `y_`-prefixed parameter names inside the model class,
   and R5's naming heuristic fired twice on lines that are not leaks. A
   trapped case must contain exactly its seeded leaks and nothing else, or
   "precision" starts measuring the benchmark author instead of the tool.
   Renamed, re-audited, locked.
**The ground truth cannot drift.** `bench/truth.py` records, per case, the
seeded leaks with family, expected rule and lines, and `tests/test_bench.py`
asserts strict equality between that registry and what the real scanner
produces on every trapped case, plus AST-silence on all 8 non-trapped cases.
"Invisible to syntax" is now a test, not a claim.
 
**Scoring conventions, stated once (`bench/score.py`).** Detections are
findings at severity ≥ medium, mirroring the tool's own reporting convention:
review-level items are questions, counted apart, never hits and never false
alarms. Matching is by line overlap, so localisation and calibration stay
separate, the severity cap can never masquerade as lost recall. Multiple
detections on one leak are all true positives; precision runs over every
detection on every file, clean ones included; control-file false positives
(clean + dependent) get their own counter; the LLM's hits on catalogue leaks
are true positives but tallied as out-of-lane, since the prompt forbids
re-reporting the rules' territory.
 
**The runner (`run_bench.py`).** Default mode is AST-only: free, offline,
deterministic, the dry run rule 9 demands before anything costly. `--llm`
adds the semantic pass (responses cached) and immediately replays the same
cached outputs with the cap disabled: one set of API calls, two
post-processings, so the ablation reports severity changes and control-file
detections with localisation held identical by construction. A final block
audits the tool's own source and `m2_backtester/` *live*, printed numbers
are measured at run time, never quoted (rule 2).
 
**Measured tonight (deterministic half, no API):** AST only, 9 detections,
9 TP, 0 FP, precision 1.00, recall 9/9 on catalogue leaks, 0/2 on semantic
ones (the gap is the point), overall recall 0.82, F1 0.90, **0 false
positives on the 6 control files**, 0 review-level questions. Live self-audit
inside the same run: 28 files, 0 findings. Suite: **222 tests** (was 187;
+26 bench, +5 rules, +5 self-audit corpus).

### Live LLM numbers - measured 2026-08-12 (night), `python run_bench.py --llm --json bench_results.json`
 
**The table.** Detection threshold: severity ≥ medium; review-level findings
are questions, counted apart.
 
| Detector | det | TP | FP | P | R catalogue (/9) | R semantic (/2) | leaks found (/11) | control-file FP (/6) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| AST only | 9 | 9 | 0 | 1.00 | 9/9 | 0/2 | 9/11 | 0 |
| LLM only (v1.2.0, cap on) | 9 | 9 | 0 | 1.00 | 7/9 | 2/2 | 9/11 | 0 |
| Hybrid (union) | 18 | 18 | 0 | 1.00 | 9/9 | 2/2 | **11/11** | 0 |
| LLM only (cap off, same cache) | 9 | 9 | 0 | 1.00 | 7/9 | 2/2 | 9/11 | 0 |
 
Plumbing: prompt 1.2.0, temperature 0; grounding rejections **0/10** findings
(no invented line in 16 files); harness caps **0**; review-level questions
**1** (dep01); cache 16 misses then 16 hits, the ablation cost nothing.
Live real-code block in the same run: this repository 28 files / 0
findings; `m2_backtester/` 16 files / 0 findings.
 
**How to read "Hybrid 18/18, F1 1.00" without overclaiming.** 18 TP is
*agreement*, not 18 discoveries: 7 leaks were found by both halves and are
counted twice by the stated convention. The headline is the last-but-one
column, **11/11 seeded leaks found, 0 false positives on 6 controls** , and
above all the *complementarity*: the deterministic rules found 9/9 syntactic
leaks and 0/2 semantic ones; the model found 2/2 semantic ones. Each half
catches exactly what the other cannot, on this benchmark. And this benchmark
is small (16 files, 11 leaks) and in-distribution by construction, we wrote
the detectors, the prompt and the cases; the clean causal engine (clean01)
resembles a worked example in the prompt. The numbers measure the design's
coherence, not its field performance. A field number needs external code
with independently established leaks; out of scope before the 16th, and
said so.
 
**Three loop-based files, same syntactic shape, three different correct
answers, all three delivered.** `clean01` (engine visible, causal) →
silence. `sem02` (engine visible, reads `prices[t+5]`) → high, self-contained.
`dep01` (no engine visible) → **review, external convention named, fix
conditional in both branches**, the model self-capped exactly as v1.2.0
asks, so the harness cap had nothing to do. This is the discrimination the
whole semantic design was built for, and it is the strongest single result of
the run.
 
**The cap did not fire, and that is the honest reading of the ablation.**
Cap-on and cap-off runs are identical (10 locations, 0 severity changes, 0
control detections either way): under v1.2.0 the model never claimed an
unearned severity on these 16 files. So the ablation is *degenerate* here, it
demonstrates compliance, not correction. Where the cap's necessity is
evidenced is the v1.1.0 episode (n=1, real code, three violations at once),
and its sufficiency is proven by the unit test that replays that real payload
(medium → review, `capped_from: medium`). Two observations of v1.2.0
compliance on dependent-type files (strategy_trend.py, dep01) cannot separate
"better prompt" from "announced enforcement"; the guarantee no longer depends
on either.
 
**The negative exclusion list is weakly obeyed, the most useful lesson for
"iterations".** The prompt says the model MUST NOT re-report the catalogue
patterns. Of 8 instances of explicitly excluded patterns in the trapped files,
the model stayed silent on 2 (`rolling(center=True)`, `raw.bfill()`) and
reported 6 (`shift(-1)`, `shift(-2)`, `fillna(method="bfill")`, both fits,
the whole-sample z-score): **exclusion compliance 2/8**. It obeyed on the two
most keyword-obvious forms and disobeyed on the same pattern in another
syntactic form (`bfill()` silent, `fillna(method="bfill")` reported). Two
consequences. First, the same asymmetry as the severity story: an instruction
is a request; lane discipline, if it matters, must be enforced in code (a
SEM finding overlapping an AST finding can be tagged "corroborates R-x" at
reporting time, never merged, per principle 4, and the corroboration
counted). Second, the LLM's 7/9 catalogue recall is not a performance number
we can interpret: on the 2 unreported cases we cannot distinguish obedience
from a miss without a lane-off prompt run. This is exactly why the AST layer
is the floor for catalogue patterns: deterministic, offline, guaranteed
recall on its territory. (Precision note: R5, feature built from the label,
is catalogue territory but is *not* on the prompt's exclusion list, so the
trap07 hit is legitimate; the runner's "7 out-of-lane" counts it, the "2/8"
above does not.)
 
**Reproducibility caveat, stated once.** The LLM rows are reproducible from
the cache (same raw outputs, deterministic post-processing), not necessarily
from fresh API calls: temperature 0 does not make a hosted model
bit-reproducible. `run_bench.py --llm --no-cache` is the experiment that
would measure run-to-run variance; not run before the deadline.
 
**Time.** Build ~2 h; live run 16 API calls, a few minutes.
 
---


## It's top secret at least for the moment
---
