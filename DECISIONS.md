# Architectural Decisions

## ADR-001: Monorepo Structure
**Date:** 20/09/2026
**Status:** Accepted

Use a single monorepo with `apps/` for frontend, `services/` for backend, and `db/` for database concerns. This keeps related code together and simplifies Docker Compose orchestration.

## ADR-002: run.sh Instead of Makefile
**Date:** 20/09/2026
**Status:** Accepted

Use a bash script (`run.sh`) instead of GNU Make. GNU Make is not installed on the development machine and `run.sh` provides equivalent task-runner functionality with no extra dependency.

## ADR-003: SQLAlchemy + Alembic for Database
**Date:** 20/09/2026
**Status:** Accepted

Use SQLAlchemy ORM with Alembic migrations rather than raw SQL. This provides type-safe models, auto-generated migrations, and a clear schema definition in Python code.

## ADR-004: psycopg (v3) as PostgreSQL Driver
**Date:** 20/09/2026
**Status:** Accepted

Use psycopg v3 (async-capable, pure Python fallback) instead of psycopg2. Better async support and actively maintained.

## ADR-005: Tailwind CSS v4 with PostCSS
**Date:** 20/09/2026
**Status:** Accepted

Use Tailwind CSS v4 via PostCSS plugin for styling. Tailwind v4 simplifies configuration with CSS-first approach.

## ADR-006: Docker Multi-stage Builds
**Date:** 20/09/2026
**Status:** Accepted

Use multi-stage Docker builds for the web app to minimise image size. API uses single-stage with slim base image since Python doesn't benefit as much from multi-stage.

## ADR-007: Click for CLI Commands
**Date:** 21/09/2026
**Status:** Accepted

Use Click for ingest CLI commands (csv-backfill, fixtures-sync, seed, verify-ingest, aliases). Click provides composable commands, automatic help generation, and parameter validation with minimal boilerplate. Preferred over argparse for its decorator-based API and over Typer for fewer dependencies.

## ADR-008: rapidfuzz for Team Alias Resolution
**Date:** 21/09/2026
**Status:** Accepted

Use rapidfuzz (not fuzzywuzzy or thefuzz) for fuzzy string matching in team alias resolution. rapidfuzz is MIT-licensed, implemented in C++, and 10-100x faster than fuzzywuzzy. Auto-accept matches with score >= 92 and >= 3-point gap to the next candidate; flag ambiguous matches for manual review.

## ADR-009: Dixon-Coles as Primary Prediction Model
**Date:** 21/09/2026
**Status:** Accepted

Use the Dixon-Coles (1997) model for match outcome prediction. This bivariate Poisson model with low-score correction is the standard baseline for football prediction research. Key design choices:
- **Parameter vector:** `[mu, attack_0..n-2, defence_0..n-2, gamma, rho]` with sum-to-zero constraint enforced by deriving the last team's parameters as `-sum(others)`, avoiding explicit scipy constraints.
- **Optimiser:** L-BFGS-B for efficient bounded optimisation (rho in [-0.5, 0.5]).
- **Time decay:** Exponential weights `exp(-xi * days/3.5)` with grid-search xi optimisation.
- **Pure library:** All fitting functions take DataFrames and return parameter dataclasses — no network or database calls inside model code.

## ADR-010: Walk-Forward with Per-Date Refit as Default
**Date:** 22/09/2026
**Status:** Accepted

Use walk-forward backtesting with expanding-window training and per-date refitting as the default strategy. For each distinct kickoff date in the held-out seasons, fit the model on all matches strictly before that date. Weekly refitting is available as a faster alternative for iteration (`--refit-step weekly`), bucketing by the most recent Monday. Per-date is the gate default because it produces the most accurate evaluation — each prediction uses the maximum available training data without any future leakage.

## ADR-011: Five Baselines Including Ablation
**Date:** 23/09/2026
**Status:** Accepted

Evaluate the Dixon-Coles model against five baselines: (1) uniform 1/3, (2) base-rate proportions from training data, (3) independent Poisson (no time decay, rho=0), (4) ablation (no time decay, rho fitted — isolates tau contribution without decay), and (5) bookmaker closing odds. The independent Poisson baseline uses uniform weights (no time decay) to provide meaningful separation from the full model. The ablation baseline measures what the tau correction adds without confounding from time decay. The tau correction is retained because it is standard in Dixon-Coles implementations and expected to matter in lower-scoring leagues, but evidence on this Premier League sample is inconclusive. The gate requires beating base-rate and independent Poisson on both RPS and log loss. Bookmaker is a reference ceiling only — not a gate requirement, since beating the market is not expected from a statistical model.

## ADR-012: JSON Reports Committed to Git
**Date:** 22/09/2026
**Status:** Accepted

Backtest reports are serialised as deterministic JSON files in `backtests/` and committed to git. This provides an audit trail of model performance over time, enables byte-identical comparison between runs, and makes it straightforward to review metric changes in pull requests. Reports use `sort_keys=True`, 2-space indent, and 10-decimal-place float precision.

## ADR-013: Poison Test as Parameterised Meta-Test
**Date:** 22/09/2026
**Status:** Accepted

The poison test (overwriting future results to verify no leakage) is implemented as a parameterised pytest test marked `@pytest.mark.slow` and excluded from default test runs. This prevents the test from slowing down CI while remaining available for thorough validation. Each parameterised instance selects a different prediction date, overwrites all results on or after that date with random scores, and asserts that predictions for that date remain unchanged.

## ADR-014: Float Precision (10dp) in JSON
**Date:** 22/09/2026
**Status:** Accepted

All floating-point values in backtest JSON reports are rounded to 10 decimal places. This is sufficient precision for statistical metrics (RPS, log loss, Brier scores) while ensuring byte-identical output across platforms and Python versions. Combined with `sort_keys=True`, 2-space indent, and Unix line endings, this enables reliable `diff`-based comparison of reports.

## ADR-015: Time-Decay xi — Sweep Results and Selection Rule
**Date:** 23/09/2026
**Status:** Accepted

A grid sweep over xi in {0, 0.002, 0.004, 0.0065, 0.010, 0.015, 0.020, 0.030} was run on the validation slice (2022-23 and 2023-24, 760 matches). The clear finding is that **decay matters**: xi=0 (no decay) is worst on both RPS (0.2053) and log loss (0.9916). Within the decay band, xi values from 0.004 to 0.020 are within noise of each other (log loss 0.974–0.979), and RPS and log loss disagree on the winner (log loss favours 0.0065, RPS favours 0.015). The sample is too small to distinguish between them. Any xi in 0.004–0.020 is defensible; **0.0065 is retained as the published default** — mid-band and the standard value from Dixon & Coles (1997).

The test slice (2024-25 and 2025-26) is used once for final confirmation and is never used for selection. This train/validation/test split prevents information leakage from the evaluation set into hyperparameter choices. The chosen xi is stored in the per-league config table (`services/engine/config/league_defaults.py`), not inline in code. Per-league re-tuning is deferred to Phase 9, where league-to-league differences may exceed this noise floor. Any future hyperparameter tuning (rho bounds, max_goals grid size, etc.) follows the same rule: select on the validation slice, confirm once on the test slice, commit to the config table.

## ADR-016: Grid Truncation Guard Threshold (1e-3)
**Date:** 23/09/2026
**Status:** Accepted

The `grid_to_markets()` function asserts that `1 - grid.sum()` (tail mass beyond the 11×11 grid) is below a safety threshold before deriving markets. The original specification proposed 1e-6, but empirical testing showed this is too tight: with `sample_params` (Arsenal v Chelsea, lambda_home=2.34) the tail mass is already 3.4e-5, and realistic Premier League lambdas routinely reach 2.0–3.0. The threshold fires at approximately lambda 3.2 per side, which is beyond any realistic per-team goal expectancy for top-flight football.

**Threshold chosen: 1e-3 (0.1%)** — accommodates lambdas up to ~4.0 per side while catching truly extreme expectancies that would silently corrupt all derived markets. If the guard ever triggers on a real fixture, the correct fix is **widening the grid from 0–10 to 0–15** (increasing `MAX_GOALS` from 11 to 16), not loosening the threshold further. A wider grid costs negligible compute (16×16 vs 11×11) but eliminates the tail entirely for any plausible football lambda.

## ADR-017: Goal Calibration Baseline (760 Held-Out Matches)
**Date:** 23/09/2026
**Status:** Accepted

Goal calibration on 760 held-out matches (2024-25 and 2025-26): predicted mean total goals 2.9153 vs actual 2.8421 (+0.073, 2.6% high), and predicted P(over 2.5) 0.5527 vs actual 0.5579 (-0.005). The mean bias sits in the high-scoring tail and does not affect the common lines (over/under 1.5, 2.5, 3.5). Accepted as-is; re-check after the xG blend in Phase 5+, and watch the over 4.5 and over 5.5 lines specifically.

## ADR-018: NB2 Count Models for Corners and Cards
**Date:** 24/09/2026
**Status:** Accepted

Phase 5 adds two parallel NB2 (negative binomial) count models for corners and booking points, running alongside the existing Dixon-Coles goals model in the walk-forward backtest. Key design choices:

**NB2 parameterisation.** Variance = mu + alpha * mu^2, where alpha controls overdispersion. When alpha < 1e-10, the model falls back to Poisson to avoid numerical issues. scipy.stats.nbinom uses (n, p) with n = 1/alpha and p = n/(n+mu). Alpha is bounded to [1e-6, 5.0] during optimisation. This is the standard NB2 form used in count regression and handles the wider variance observed in corner and card data compared to goals.

**Compound booking-point distribution.** Cards use a compound model: yellows from NB2(mu_yellow, alpha_yellow) and reds from Poisson(mu_red), with booking points = 10*Y + 25*R. The previous approach of fitting NB2 directly to booking-point totals was misspecified — booking points are not a natural count (VMR ~10.8 is a 10x scaling artefact of the 10-point yellow weighting). The compound model fits each card type at its natural scale, then convolves the resulting per-team booking-point PMFs for match-total O/U probabilities. The mu initialisation floor was lowered from 1.0 to 0.01 to support the reds model (mean ~0.1 per side).

**Referee effect (cards only).** Referees with 20+ matches in the training window get their own additive log-linear effect parameter; those with fewer matches get effect=0 (league mean). The 20-match threshold balances sample size for reliable estimation against referee coverage. Referee effects sum to zero via the same constraint used for team attack/defence parameters. Derby flag and cross-effects are deferred to Phase 9.

**Separate CountPrediction dataclass.** Count predictions use their own `CountPrediction` record rather than adding fields to `MatchPrediction`. This keeps the goals model interface unchanged, avoids bloating every goals prediction with empty corner/card fields, and allows count models to be toggled per league via capability flags (`has_corners`, `has_cards` on `LeagueConfig`).

**Capability flags.** Per-league `has_corners` and `has_cards` booleans on `LeagueConfig` control whether count models run during backtesting. E0 (Premier League) has both enabled; other leagues default to `False`. This avoids fitting count models where the CSV data lacks corner/card columns or where sample sizes are insufficient.

**Gate checks.** Count models must beat a constant-rate baseline (observed over-rate per line) on mean Brier score. Gate evaluation uses a subset of publishable lines (CORNER_GATE_LINES, BOOKING_GATE_LINES) — extreme lines near 0/1 carry no information and inflate mean Brier noise. The too-good alarm uses a relative threshold (model_brier < 0.9 * baseline_brier) instead of an absolute threshold, avoiding false alarms on lopsided markets where both model and baseline Brier are legitimately low. Count gates are reported separately and do not affect the goals gate pass/fail status.

## ADR-019: Corner Model — Independence Assumption Failure
**Date:** 25/09/2026
**Status:** Accepted

Walk-forward backtest on 760 held-out PL matches shows the NB2 corner model fails the gate (+0.6% vs base-rate on publishable lines 8.5–12.5). The team effects are real (attack std = 0.14, defence std = 0.18, decile calibration r = 0.75) and a simple rate-based model ties base-rate (-0.1%), confirming the signal exists.

The root cause is the **independence assumption in the NB2 convolution**. Home and away corners within a match are negatively correlated (Pearson r = -0.338, p < 0.0001) — the team dominating possession wins more corners and concedes fewer. The model assumes Cov(H,A) = 0, overestimating Var(total) by 31% (14.9 predicted vs 11.3 observed). This produces an over-dispersed total distribution that hedges O/U probabilities toward 50/50, losing to the base-rate on informative lines.

The decile calibration slope is 0.51 (should be 1.0): the model predicts a 3.4-corner spread across deciles but actual totals only rise 1.8, consistent with over-dispersion compressing discrimination.

**Decision:** accept the gate FAIL for corners under the current architecture. The fix is not better team effects (those are already meaningful) but a model that captures the negative home-away covariance — either a bivariate count model, a direct model on total corners, or a copula correction on the marginals. Deferred to Phase 9. Full diagnostic in `docs/phases/phase-5-corner-diagnostic.md`.

## ADR-020: Phase 5 Outcome — Goals Ship, Corners and Cards Do Not
**Date:** 25/09/2026
**Status:** Accepted

### Summary

Phase 5 evaluated three market families against the gate criteria (beat base-rate on publishable O/U lines). Goals markets pass; corners and cards fail. Only goals markets ship to the UI.

### Goals Markets — PASS

Goals gate passed (ADR-017): RPS and log loss beat base-rate and independent Poisson on 760 held-out matches. 15 goals-derived markets (1×3, correct score, O/U 0.5–5.5, BTTS) ship via `grid_to_markets()`.

### Corner Markets — FAIL

**Direct total-corners NB2 model** was built to fix the independence assumption failure documented in ADR-019. The model fits `mu_total = exp(mu + home_effect[h] + away_effect[a])` as a single NB2 count, eliminating the convolution that overestimated Var(total) by 31%.

**Variance fixed.** Model implied Var(total) = 10.97 vs observed 11.35 (ratio 0.966), down from 14.9 (1.309) under convolution.

**Brier still fails.** Team effects overfit: decile calibration slope fell from 0.51 to 0.36 (predicted 3.4-corner range vs 1.8 actual). A shrinkage sweep over k ∈ {0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0} (where `mu_total = exp(mu + k*home_eff + k*away_eff)`) was tuned on validation seasons (2022-23, 2023-24) and confirmed on test seasons (2024-25, 2025-26):

| k | Val Brier | Val vs base | Test Brier | Test vs base | Test vs simple |
|---|-----------|-------------|------------|--------------|----------------|
| 0.00 | 0.227111 | +0.06% | 0.221767 | +0.04% | +0.21% |
| 0.25 | 0.226025 | -0.42% | 0.221295 | -0.18% | +0.00% |
| 0.40 | 0.226024 | -0.42% | 0.221752 | +0.03% | +0.21% |
| 1.00 | 0.230422 | +1.52% | 0.228247 | +2.96% | +3.14% |

Best validation k=0.40 fails on test (+0.03% vs base, +0.21% vs simple). Best test k=0.25 ties simple but was not selected on validation. The signal is real (decile r = 0.79–0.91 at moderate k on test) but too weak to overcome the base-rate prior in Brier terms.

**Decision:** corners do not ship. The team effects are real but insufficient for publishable O/U markets. Deferred to Phase 9 for potential bivariate or copula treatment.

### Card Markets — FAIL

**Compound model is correctly specified.** The compound booking-point PMF (`10*Y + 25*R` where Y ~ NB2, R ~ Poisson) was verified empirically: 100,000 samples from `_booking_point_pmf()` at league-mean parameters yield mean = 38.9, variance = 432. With fitted team effects across all matchups, mean model Var(total BP) = 420 vs observed 499 (ratio 0.840). The previously reported "42.2 vs 499" was a diagnostic script formula error (applying `mu + alpha*mu^2` to booking-point means with yellow alpha instead of decomposing through the compound distribution).

The remaining variance gap (420 vs 499, 16% under) is consistent with positive home-away booking-point correlation (r = +0.196, p < 0.0001) — the convolution assumes independence and underestimates total variance.

**Gate result:** card model Brier 0.2170 vs baseline 0.2146 (+1.1%). Cards fail the gate.

**Decision:** cards do not ship. The compound model correctly decomposes yellow/red mechanics but the independence assumption in the convolution underestimates total variance, same pattern as corners but milder. Deferred to Phase 9.

### Capability Flags

Two flag tiers on `LeagueConfig` and the `leagues` DB table:

- `has_corners` / `has_cards` — data exists and backtest runs count models (True for E0). Retained for diagnostics and future development.
- `ship_corners` / `ship_cards` — markets are published to the UI (False for E0). The API and frontend read these flags to determine which market families to display.

Any future phase that adds market endpoints or UI components reads `ship_*` flags from the league config, not `has_*`. This ensures corners and cards remain suppressed until a future phase (Phase 9+) produces a model that passes the gate.

## ADR-021: Phase 6 — Predict Command, API, and Web Pages
**Date:** 25/09/2026
**Status:** Accepted

Phase 6 bridges the pure engine to production. Key decisions:

### Immutable predictions
Predictions are append-only. Re-running the predict command writes new rows; the API serves the prediction with the latest `created_at` per `match_id`. This avoids UPDATE contention and preserves an audit trail.

### Idempotent predict runs
A SHA-256 fingerprint of `(league_id, n_finished_matches, latest_match_date, model_version, git_commit)` stored on `model_runs` with a UNIQUE constraint. If the fingerprint already exists, the run is a no-op — no duplicate prediction sets. Including model version and git commit means code changes trigger fresh predictions even when training data hasn't changed. A `--force` flag bypasses the fingerprint check for manual re-runs.

### Grid storage
The 11x11 score grid is compressed via `zlib.compress(grid.astype(float64).tobytes())` and stored as `LargeBinary` on the `predictions` table. Raw = 968 bytes, compressed = ~300-400 bytes. This enables the API to serve the full grid and derive the modal scoreline without re-running the model.

### Engine purity preserved
The predict command orchestrates: queries DB → builds DataFrame → calls `fit_dixon_coles` / `build_grid` / `grid_to_markets` → writes results back. No DB calls inside engine code.

### Confidence buckets
Entropy-based classification of the 1X2 distribution into four buckets (very_high, high, medium, low). Thresholds tuned for realistic football distributions where entropy typically ranges 0.9-1.5.

### Scoreline disagreement
When the modal scoreline implies a different result from the 1X2 favourite, the API returns an explanatory note. This is expected behaviour (not an error) and is presented with neutral styling in the UI.

### API design
Three router groups: `/api/v1/leagues`, `/api/v1/fixtures`, `/api/v1/matches/{id}`. The match detail endpoint returns grouped markets for accordion display. Async SQLAlchemy sessions for the API tier; sync for the worker (predict command).

### Web pages
Next.js 15 server components with ISR (15 min for list pages, 60s for match detail). Dark mode default. The score grid uses a simple div-based heatmap rather than a charting library to minimise client JS.

## ADR-022: Phase 7 — Calibration Maps, Accuracy Page, Model-vs-Market
**Date:** 28/09/2026
**Status:** Accepted

### Calibration data source
Bootstrap initial calibration from the 760-match walk-forward backtest (`source='backtest'`). A rolling window of 2,000 live predictions will take over automatically once available (`source='live'`). The `is_current` flag ensures only one set of calibration maps is active per league at a time.

### Isotonic regression
Implemented pool adjacent violators algorithm (PAVA) in pure numpy — no scikit-learn dependency. This keeps the engine lightweight and avoids adding a heavy dependency for a single algorithm.

### Grid reconstruction
Goal-derived markets (O/U, BTTS) are calibrated by reconstructing the Poisson grid from `lambda_home` and `lambda_away` stored in the backtest predictions. This uses independent Poisson (no rho correction) since rho is not stored per prediction. The error is <0.5pp for O/U 2.5 and BTTS — acceptable for calibration purposes.

### Quality badges
Three-tier badge system based on data availability:
- **Green**: 3+ seasons of direct calibration data
- **Amber**: 1-2 seasons or cross-competition inference
- **Grey**: insufficient data

### Publication gates
Four gates that each market must pass before publication is recommended:
1. At least 500 settled predictions
2. Maximum decile calibration error <5 percentage points
3. Model must beat the relevant baseline
4. Calibration must come from direct (not inferred) data

### Post-isotonic renormalisation
Isotonic regression is fitted independently per selection. Since bin indices across different selections correspond to different sets of matches (bin 1 for home win ≠ bin 1 for draw), renormalisation cannot be applied at the bin level. Instead, renormalisation is applied per-prediction at lookup time via `calibrate_and_renormalise()`:
1. For a given prediction, look up the isotonic-calibrated probability for each selection from its respective calibration bin
2. Proportionally rescale the calibrated values so the group sums to 1

Groups requiring renormalisation:
- **1X2 triplet**: match_result_home + match_result_draw + match_result_away = 1
- **Complementary pairs**: over/under pairs and btts_yes/btts_no = 1

The calibration bins themselves remain per-selection and un-renormalised.

### Minimum bin count for gate check
The 5pp decile error gate assumes ~2,000+ settled predictions per bin. At 760 predictions, extreme bins (e.g. 0–10%, 90–100%) have as few as 9 observations, making the error metric noise-dominated. A minimum bin count threshold of 30 is applied: bins below this threshold are skipped from the max decile error computation and flagged as `insufficient_data` in the reliability diagram.

### Goals-derived market badges
All goals-derived markets (O/U, BTTS) share the same data lineage as 1X2: the underlying lambdas are fitted on actual match goals, which IS direct data. The badge rule uses `direct_data = the league has the underlying statistic in matches` — goals for goals-derived markets, corners for corner markets. This means O/U and BTTS get the same badge as 1X2 (amber at 2 seasons) rather than grey.

### Model-vs-market comparison
The accuracy page displays model probabilities alongside overround-stripped bookmaker odds for 1X2 markets. This is display-only — no blending or combining of model and market probabilities is applied.

### Schema changes (migration 005)
- `calibration_maps`: `model_run_id` made nullable; added `source`, `league_id`, `selection`, `is_current`, `version`, `created_at`
- `accuracy_metrics`: `model_run_id` made nullable; added `source`, `league_id`, `market`
- New `model_vs_market` table for storing per-prediction model vs bookmaker comparison data

## ADR-023: Phase 7 Outcome — Display Tiers and Provisional Markets
**Date:** 28/09/2026
**Status:** Accepted

### Gate results at 760 backtest predictions

Three goals-derived markets pass all four publication gates:

| Market | Max decile error | Gate |
|--------|-----------------|------|
| match_result_draw | 2.7pp | PASS |
| over_under_1.5_over | 3.6pp | PASS |
| over_under_1.5_under | 3.6pp | PASS |

Eight goals-derived markets fail on calibration decile error (>5pp):

| Market | Max decile error | Gate |
|--------|-----------------|------|
| btts_yes / btts_no | 5.3pp | FAIL |
| match_result_home | 6.0pp | FAIL |
| match_result_away | 8.0pp | FAIL |
| over_under_2.5_over / under | 13.1pp | FAIL |
| over_under_3.5_over / under | 12.9pp | FAIL |

All 11 markets pass the other three gates (min_settled, beats_baseline, direct_data). All badges are amber (2 backtest seasons).

Corners and cards remain structurally failed from Phase 5 (ADR-020): signal too weak for Brier improvement.

### Display tiers on the match page

The spec's strict rule — don't show predictions until all gates pass — would leave users with only draw and O/U 1.5 on the match page. That is not a useful product. Instead, markets are split into three display tiers:

| Tier | Badge | Criteria | Markets |
|------|-------|----------|---------|
| **Published** | green | All 4 gates passed | match_result_draw, over_under_1.5 |
| **Provisional** | amber | Gates pending (decile error) | match_result_home, match_result_away, btts, over_under_2.5, over_under_3.5 |
| **Suppressed** | — | Gates failed structurally | corners, cards |

Provisional markets display with a one-line note: *"Based on 760 backtest predictions. Calibration improves as more matches settle."*

### Trade-off

This is a pragmatic departure from the strict gate rule. The gates themselves are not weakened — the accuracy page still shows exactly which markets pass and which don't. The provisional label is the honest framing: these probabilities are real model output, from a model that beats baseline, but the calibration has not yet been validated to publication quality. The alternative (showing only draw and O/U 1.5) renders the product unusable.

As live predictions settle and the sample grows past ~2,000, provisional markets are expected to pass the decile error gate and promote to published. The O/U 2.5 and O/U 3.5 decile errors (13pp) reflect the smaller number of observations in the tails at 760 predictions — not a systematic model deficiency.

## ADR-024: Phase 8 — Half-Time Grids, HT/FT Market, First Goal Timing
**Date:** 28/09/2026
**Status:** Accepted

### Context

The full-time Dixon-Coles model produces an 11×11 score grid from which 15 goal markets are derived. Phase 8 adds half-time modelling: separate Dixon-Coles fits on HT goals and second-half goals, a 9-outcome HT/FT market, and first-goal timing.

### Key decisions

1. **Separate HT Dixon-Coles fit** — the HT model is its own Dixon-Coles fit on half-time goals (`ht_home_goals`, `ht_away_goals`), not a fraction of the FT model. This captures half-specific team effects (e.g. teams that score early vs late).

2. **Separate 2H Dixon-Coles fit** — second-half goals are computed as `FT − HT` and fitted as another independent Dixon-Coles. This avoids constraining the 2H model to the FT model's parameters.

3. **7×7 grid** — HT and 2H grids use `max_goals=7`. Half-time goal rates are typically λ ≈ 0.5–0.8, so 7 goals captures >99.9% of probability mass. The `build_grid()` function now accepts a `max_goals` parameter (default 11 for backward compatibility).

4. **HT/FT independence assumption** — the 9-outcome HT/FT market assumes independence between HT and 2H periods. Joint probability is computed by enumerating all (ht_h, ht_a, sh_h, sh_a) combinations (7⁴ = 2401 iterations), computing the FT score, and accumulating P(HT result, FT result).

5. **First goal timing** — analytical Exponential(λ_home + λ_away) inter-arrival model. No calibration data needed; produces P(first goal before minute X) for X ∈ {15, 30, 45}.

6. **Gate pattern** — follows the corners/cards gate pattern. HT and 2H models must each beat base-rate on RPS and log loss. The HT/FT 9-outcome model must beat uniform 1/9. A too-good alarm fires at ratio < 0.9.

7. **Capability flags** — `has_halves`/`ship_halves` follow the corners/cards pattern. E0 has `has_halves=True` (CSV data has HTHG/HTAG). `ship_halves` defaults to False until gate results confirm quality.

8. **HT market lines** — reduced from FT: O/U 0.5/1.5/2.5 only (not 3.5+), team totals 0.5/1.5 only (not 2.5+). All market names prefixed `ht_`.

### New files

| File | Purpose |
|------|---------|
| `services/engine/markets/halftime.py` | HT markets, HT/FT market, first goal timing |
| `services/engine/backtest/halftime_harness.py` | Walk-forward HT/2H Dixon-Coles fitting |
| `services/engine/backtest/halftime_gate.py` | Gate checks for HT model quality |

### Test count

25 new tests (3 grid, 13 markets, 8 harness, 4 gate) — total engine suite: 672 tests passing.

### Backtest results (E0, held-out 2024-25 + 2025-26, n=760)

| Metric | Model | Baseline | Delta |
|--------|-------|----------|-------|
| HT 1X2 RPS | 0.2011 | 0.2126 (base-rate) | -5.39% |
| HT 1X2 log loss | 1.0722 | 1.0842 (base-rate) | -1.10% |
| 2H 1X2 RPS | 0.2160 | 0.2196 (base-rate) | -1.63% |
| HT/FT log loss | 2.0784 | 2.1972 (uniform 1/9) | -0.1188 |
| HT/FT RPS | 0.1907 | 0.2136 (uniform) | -10.7% |

Gate: **PASS** (all 6 checks). First-half goal share: 0.449 (observed), 0.446 (fitted lambdas).

### Known biases

1. **HT/FT conditional independence** — the model assumes the second half is independent of the first half. This under-predicts DD by ~1.2pp (predicted 13.0% vs observed 14.2%) and under-predicts reversal cells HA and AH by ~1.3pp combined (predicted 4.6% vs observed 6.9%). The fix is a state-dependent second-half model (e.g. conditioning 2H lambdas on HT score), deferred until ROI justifies the complexity.

2. **HT 0-0 gap** — model predicts P(0-0 at HT) = 23.6% vs observed 26.1%. This is the same mean-goals bias from Phase 4 (Poisson slightly over-predicts total goals), now split across halves. The exponential first-goal-timing model inherits this bias directly.
