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
