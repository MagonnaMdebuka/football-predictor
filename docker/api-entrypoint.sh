#!/bin/bash
set -e

echo "Running database migrations..."
cd /app
alembic -c db/alembic.ini upgrade head

echo "Starting API server..."
exec uvicorn services.api.main:app --host 0.0.0.0 --port 8000 "$@"
