#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run-job.sh — cron wrapper for football-predictor scheduled jobs
#
# Usage:  run-job.sh <job_name> "<command inside worker container>"
#
# What it does:
#   1. Runs the command inside the worker container via docker compose exec
#   2. Appends stdout/stderr to /var/log/football-predictor/<job>.log
#   3. Records the exit code to a state file for consecutive-failure tracking
#   4. If the last TWO runs both failed, posts to ALERT_WEBHOOK_URL
# ---------------------------------------------------------------------------
set -euo pipefail

JOB_NAME="${1:?Usage: run-job.sh <job_name> <command>}"
shift
COMMAND="$*"

PROJECT_DIR="${PROJECT_DIR:-/opt/football-predictor}"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_DIR/docker-compose.prod.yml}"
LOG_DIR="/var/log/football-predictor"
STATE_DIR="/var/lib/football-predictor/state"

mkdir -p "$LOG_DIR" "$STATE_DIR"

LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
STATE_FILE="$STATE_DIR/${JOB_NAME}.last_exit"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------------------
# Run the job
# ---------------------------------------------------------------------------
echo "[$TIMESTAMP] Starting job: $JOB_NAME" >> "$LOG_FILE"

EXIT_CODE=0
docker compose -f "$COMPOSE_FILE" exec -T worker $COMMAND >> "$LOG_FILE" 2>&1 || EXIT_CODE=$?

END_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[$END_TIMESTAMP] Finished job: $JOB_NAME (exit=$EXIT_CODE)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ---------------------------------------------------------------------------
# Track consecutive failures
# ---------------------------------------------------------------------------
# State file stores the previous exit code (one line).
PREV_EXIT=0
if [[ -f "$STATE_FILE" ]]; then
    PREV_EXIT=$(cat "$STATE_FILE" 2>/dev/null || echo "0")
fi

# Write current exit code for next run
echo "$EXIT_CODE" > "$STATE_FILE"

# ---------------------------------------------------------------------------
# Alert on two consecutive failures
# ---------------------------------------------------------------------------
if [[ "$EXIT_CODE" -ne 0 && "$PREV_EXIT" -ne 0 ]]; then
    # Source the .env to pick up ALERT_WEBHOOK_URL
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        ALERT_WEBHOOK_URL=$(grep -oP '^ALERT_WEBHOOK_URL=\K.*' "$PROJECT_DIR/.env" || true)
    fi

    if [[ -n "${ALERT_WEBHOOK_URL:-}" ]]; then
        PAYLOAD=$(cat <<ENDJSON
{
  "job": "$JOB_NAME",
  "status": "two_consecutive_failures",
  "current_exit": $EXIT_CODE,
  "previous_exit": $PREV_EXIT,
  "timestamp": "$END_TIMESTAMP",
  "host": "$(hostname)"
}
ENDJSON
)
        curl -sf -X POST \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" \
            "$ALERT_WEBHOOK_URL" >> "$LOG_FILE" 2>&1 || true

        echo "[$END_TIMESTAMP] ALERT sent for $JOB_NAME (two consecutive failures)" >> "$LOG_FILE"
    else
        echo "[$END_TIMESTAMP] WARNING: $JOB_NAME has two consecutive failures but ALERT_WEBHOOK_URL is not set" >> "$LOG_FILE"
    fi
fi

exit "$EXIT_CODE"
