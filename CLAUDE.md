# Football Predictor — Claude Code Instructions

## Never re-run a completed backtest
If a diagnostic script crashes after the backtest finishes, extract the data
that already printed and compute the remaining sections from it. A script
that calls run_backtest() is never the fix for a missing print statement.

## Current Phase
Phase 9 (Multi-League Backfill, Cron, Deploy, Monitoring) — planning.
Phase 8 (Half-Time Grids, HT/FT Market, First Goal Timing) — complete (ADR-024). 672+ tests.
Phase 7 (Calibration, Accuracy, Quality Badges) — complete (ADR-022).
Phase 6 (Predict, API, Web) — complete (ADR-021).
Phase 5 (Corners & Cards) — complete (ADR-020).
Phase 4 (Markets) — complete.
Phase 3 (Backtest) — complete.
Phase 2 (Engine) — complete.
Phase 1 (Ingest) — complete.

## Project Layout
```
apps/web/              Next.js 15 frontend
services/api/          FastAPI read API
services/engine/       ingest/, models/, markets/, backtest/, predict/, calibration/
db/migrations/         Alembic (PostgreSQL 16)
docker-compose.yml     5 services: db, redis, api, web, worker
run.sh                 Task runner (replaces Makefile)
```

## Key Conventions
- Python 3.12, type hints throughout, ruff for linting (line-length 100)
- All timestamps stored as UTC in the database
- Engine is a pure library: fitting functions take DataFrames and return parameters — no network or DB calls inside model code
- Sync SQLAlchemy for the worker; async for the API
- Upserts via `INSERT ... ON CONFLICT DO UPDATE` on natural keys for idempotency
- Every ingest run records to the `ingest_runs` table with accurate counts

## Testing
- Write the test before the implementation for anything in `services/engine/`
- `pytest` with `-v` flag; tests live in `tests/` mirroring `services/` structure
- Test fixtures in `tests/fixtures/`

## Data Sources
- football-data.co.uk CSVs — primary training data, cached to `data/raw/`
- football-data.org v4 API — fixtures and results (10 req/min, token required)
- Rate-limit rule: scheduled workers write to Postgres; web tier reads only

## Run Commands
```bash
./run.sh dev              # Start all services
./run.sh test             # Run pytest
./run.sh test-models      # Run Dixon-Coles model tests only
./run.sh test-backtest    # Run backtest test suite only
./run.sh predict          # Run predictions for a league
./run.sh backtest         # Run walk-forward backtest
./run.sh backtest-summary # Print summary of a backtest JSON report
./run.sh backtest-compare # Compare two reports for byte-identical output
./run.sh calibrate        # Bootstrap calibration from backtest
./run.sh csv-backfill     # Backfill CSV data (--league E0, --offline)
./run.sh fixtures-sync    # Sync upcoming fixtures (--league E0 or all)
./run.sh seed             # Seed leagues, seasons, aliases
./run.sh verify-ingest    # Run quality checks (--league E0 or all)
./run.sh aliases          # Review/confirm team aliases
./run.sh daily            # fixtures-sync + predict --all-active
```
