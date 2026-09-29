#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh — deploy football-predictor to the production VPS
#
# Usage:
#   ./deploy.sh                    # deploy using defaults
#   ./deploy.sh --host 1.2.3.4    # override host
#   ./deploy.sh --dry-run          # show what would run without executing
#
# Prerequisites:
#   - SSH key access to the VPS (ssh-agent or ~/.ssh/config)
#   - Docker and Docker Compose installed on the VPS
#   - Repository cloned to /opt/football-predictor on the VPS
#   - .env file configured on the VPS with production secrets
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
REMOTE_HOST="${DEPLOY_HOST:-}"
REMOTE_USER="${DEPLOY_USER:-root}"
REMOTE_DIR="/opt/football-predictor"
BRANCH="main"
DRY_RUN=false
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)     REMOTE_HOST="$2"; shift 2 ;;
        --user)     REMOTE_USER="$2"; shift 2 ;;
        --branch)   BRANCH="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: ./deploy.sh [--host HOST] [--user USER] [--branch BRANCH] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --host HOST      VPS hostname or IP (or set DEPLOY_HOST env var)"
            echo "  --user USER      SSH user (default: root, or set DEPLOY_USER)"
            echo "  --branch BRANCH  Git branch to deploy (default: main)"
            echo "  --dry-run        Print commands without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    echo "Error: no host specified. Use --host or set DEPLOY_HOST." >&2
    exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
COMPOSE_FILE="docker-compose.prod.yml"

# ── Helpers ───────────────────────────────────────────────────────────
log()  { echo "==> $*"; }
run()  {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

remote() {
    # Execute a command on the VPS via SSH
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] ssh $SSH_OPTS $REMOTE \"$*\""
    else
        ssh $SSH_OPTS "$REMOTE" "$*"
    fi
}

# ── Pre-flight checks ────────────────────────────────────────────────
log "Deploying branch '$BRANCH' to $REMOTE:$REMOTE_DIR"

log "Checking SSH connectivity..."
run ssh $SSH_OPTS "$REMOTE" "echo 'SSH OK'" || {
    echo "Error: cannot reach $REMOTE via SSH" >&2
    exit 1
}

log "Checking remote prerequisites..."
remote "test -d $REMOTE_DIR/.git" || {
    echo "Error: $REMOTE_DIR is not a git repository on the VPS." >&2
    echo "Clone it first:  git clone <repo-url> $REMOTE_DIR" >&2
    exit 1
}
remote "test -f $REMOTE_DIR/.env" || {
    echo "Error: $REMOTE_DIR/.env not found on VPS. Create it from .env.example." >&2
    exit 1
}

# ── Step 1: Pull latest code ─────────────────────────────────────────
log "Pulling latest code (branch: $BRANCH)..."
remote "cd $REMOTE_DIR && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"

# ── Step 2: Build images ─────────────────────────────────────────────
log "Building Docker images..."
remote "cd $REMOTE_DIR && docker compose -f $COMPOSE_FILE build"

# ── Step 3: Run database migrations ──────────────────────────────────
log "Running database migrations..."
# Start only db so migrations can run, then the API entrypoint runs them
# on startup too — but running them explicitly first catches errors early.
remote "cd $REMOTE_DIR && docker compose -f $COMPOSE_FILE up -d db"
remote "cd $REMOTE_DIR && docker compose -f $COMPOSE_FILE run --rm api alembic -c db/alembic.ini upgrade head"

# ── Step 4: Restart services ─────────────────────────────────────────
log "Restarting services..."
remote "cd $REMOTE_DIR && docker compose -f $COMPOSE_FILE up -d"

# ── Step 5: Wait for health checks ───────────────────────────────────
log "Waiting for services to become healthy..."
HEALTH_TIMEOUT=60
HEALTH_OK=false

for i in $(seq 1 $HEALTH_TIMEOUT); do
    if [[ "$DRY_RUN" == true ]]; then
        HEALTH_OK=true
        break
    fi
    STATUS=$(ssh $SSH_OPTS "$REMOTE" \
        "curl -sf http://localhost:8000/health 2>/dev/null | python3 -c \"import sys,json; print(json.load(sys.stdin).get('status',''))\" 2>/dev/null || echo 'down'")
    if [[ "$STATUS" == "healthy" ]]; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if [[ "$HEALTH_OK" == true ]]; then
    log "Health check passed"
else
    echo "Warning: API did not become healthy within ${HEALTH_TIMEOUT}s" >&2
    echo "Check logs: ssh $REMOTE 'cd $REMOTE_DIR && docker compose -f $COMPOSE_FILE logs api'" >&2
fi

# ── Step 6: Install/update crontab ────────────────────────────────────
log "Installing crontab..."
remote "cp $REMOTE_DIR/cron/football-predictor.crontab /etc/cron.d/football-predictor && chmod 644 /etc/cron.d/football-predictor"
remote "chmod +x $REMOTE_DIR/cron/run-job.sh"
remote "mkdir -p /var/log/football-predictor /var/lib/football-predictor/state"

# ── Step 7: Clean up old images ───────────────────────────────────────
log "Pruning unused Docker images..."
remote "docker image prune -f --filter 'until=168h'" || true

# ── Done ──────────────────────────────────────────────────────────────
DEPLOYED_COMMIT=$(ssh $SSH_OPTS "$REMOTE" "cd $REMOTE_DIR && git rev-parse --short HEAD" 2>/dev/null || echo "unknown")
log "Deploy complete — commit $DEPLOYED_COMMIT on $REMOTE_HOST"
