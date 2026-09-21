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
