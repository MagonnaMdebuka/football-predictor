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

## ADR-011: Four Baselines Including Independent Poisson
**Date:** 22/09/2026
**Status:** Accepted

Evaluate the Dixon-Coles model against four baselines: (1) uniform 1/3, (2) base-rate proportions from training data, (3) independent Poisson (Dixon-Coles with rho=0), and (4) bookmaker closing odds. The independent Poisson baseline is a diagnostic tool that isolates the contribution of the tau correction and time decay. The gate requires beating base-rate and independent Poisson on both RPS and log loss. Bookmaker is a reference ceiling only — not a gate requirement, since beating the market is not expected from a statistical model.

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
