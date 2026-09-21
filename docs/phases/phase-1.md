# Phase 1: Ingest

Build the full ingestion layer for the Premier League: CSV backfill from football-data.co.uk
(7 completed seasons + current), upcoming fixtures from football-data.org, team alias resolution,
quality checks, and CLI commands. No models, predictions, API endpoints, or UI in this phase.

## Scope

### Data Sources
1. **football-data.co.uk** — Season CSVs for Premier League (division E0), seasons 2019-20
   through 2026-27. Columns: Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG, HS, AS,
   HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, Referee.
2. **football-data.org v4** — Upcoming fixtures, match results. 10 requests per minute,
   free-tier token required.

### Components
| File | Purpose |
|------|---------|
| `config.py` | `IngestConfig` from env vars (token, URLs, thresholds) |
| `constants.py` | CSV column map, season codes, no-crowd date range |
| `normalise.py` | Text normalisation, timezone conversion (Europe/London to UTC), season helpers |
| `db_session.py` | Sync SQLAlchemy session factory |
| `http_client.py` | `RateLimitedClient` — token bucket (10/min), backoff+jitter, circuit breaker, Redis daily budget |
| `alias_resolver.py` | Exact, normalised, then rapidfuzz resolution; auto-accept >= 92 with >= 3 gap |
| `seed.py` | Seed leagues, seasons, team aliases from CSV |
| `csv_parser.py` | Parse fd.co.uk CSV: utf-8-sig/latin-1, date formats, column mapping, kickoff UTC |
| `csv_loader.py` | Download, cache, parse, resolve, upsert orchestration |
| `fixtures_loader.py` | football-data.org v4 fixture sync |
| `quality.py` | Per-season validation (380 matches, 20 teams, stat consistency) |
| `cli.py` | Click commands: csv-backfill, fixtures-sync, seed, verify-ingest, aliases review/confirm |
| `runner.py` | Wiring layer connecting CLI to components |

### Schema Changes (Migration 002)
- `leagues`: add `fd_couk_code`, `fd_org_code`, `has_corners`, `has_cards`, `has_xg`
- `teams`: rename `name` to `canonical_name`
- `team_aliases`: make `team_id` nullable, add `score`, `confirmed`
- `matches`: rename goals columns, add 16 stat columns, change unique constraint
- `ingest_runs`: rename `records_processed` to `rows_written`, add `job`, `rows_skipped`
- `match_source_rows`: new table for raw source data

### CLI Commands
```bash
./run.sh seed             # Seed leagues, seasons, team aliases
./run.sh csv-backfill     # Download and load all season CSVs
./run.sh fixtures-sync    # Sync upcoming fixtures from football-data.org
./run.sh verify-ingest    # Run quality checks on ingested data
./run.sh aliases          # Review/confirm unconfirmed team aliases
```

## Gate Criteria
1. 2,660+ completed matches for 2019-20 to 2025-26, plus 2026-27 to date
2. Zero unresolved aliases — all team names resolved
3. Next 14 days of fixtures linked to the same teams
4. Every job re-run writes zero changes (idempotency)
5. All tests pass
6. `ingest_runs` row with accurate counts for every run
7. Paste `./run.sh verify-ingest` output, update CLAUDE.md phase line, stop
