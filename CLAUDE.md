# Football Predictor — Claude Code Instructions

## Current Phase
Phase 1 (Ingest) — in progress.

## Project Layout
```
apps/web/              Next.js 15 frontend
services/api/          FastAPI read API
services/engine/       ingest/, models/, markets/, backtest/
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
./run.sh csv-backfill     # Backfill CSV data
./run.sh fixtures-sync    # Sync upcoming fixtures
./run.sh seed             # Seed leagues, seasons, aliases
./run.sh verify-ingest    # Run quality checks
./run.sh aliases          # Review/confirm team aliases
```
