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
