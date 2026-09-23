#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Copy .env from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

case "${1:-help}" in
    dev)
        echo "Starting all services..."
        docker compose up --build -d
        echo ""
        echo "Waiting for services to become healthy..."
        echo "  API:  http://localhost:8000/health"
        echo "  Web:  http://localhost:3000/health"
        echo ""
        echo "Run './run.sh logs' to follow logs"
        echo "Run './run.sh verify' to check health"
        ;;

    stop)
        echo "Stopping all services..."
        docker compose down
        ;;

    logs)
        docker compose logs -f "${@:2}"
        ;;

    health)
        echo "=== API Health ==="
        curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "API not responding"
        echo ""
        echo "=== Web Health ==="
        curl -s http://localhost:3000/health | python -m json.tool 2>/dev/null || echo "Web not responding"
        ;;

    verify)
        PASS=true

        echo "Checking API health..."
        API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
        if [ "$API_STATUS" = "200" ]; then
            echo "  API: PASS (HTTP $API_STATUS)"
        else
            echo "  API: FAIL (HTTP $API_STATUS)"
            PASS=false
        fi

        echo "Checking Web health..."
        WEB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/health 2>/dev/null || echo "000")
        if [ "$WEB_STATUS" = "200" ]; then
            echo "  Web: PASS (HTTP $WEB_STATUS)"
        else
            echo "  Web: FAIL (HTTP $WEB_STATUS)"
            PASS=false
        fi

        echo ""
        if [ "$PASS" = true ]; then
            echo "Gate: PASS — Phase 0 complete"
            exit 0
        else
            echo "Gate: FAIL — services not healthy"
            exit 1
        fi
        ;;

    lint)
        echo "Linting Python (ruff)..."
        docker compose exec api ruff check /app/services/api /app/db || true
        echo ""
        echo "Linting TypeScript (next lint)..."
        docker compose exec web npx next lint || true
        ;;

    test)
        echo "Running API tests..."
        docker compose exec api python -m pytest /app/tests/api -v
        echo ""
        echo "Running engine tests..."
        docker compose exec worker python -m pytest /app/tests/engine -v
        ;;

    test-models)
        echo "Running Dixon-Coles model tests..."
        docker compose exec worker python -m pytest /app/tests/engine/models -v
        ;;

    test-backtest)
        echo "Running backtest tests..."
        docker compose exec worker python -m pytest /app/tests/engine/backtest -v -m "not slow"
        ;;

    backtest)
        echo "Running walk-forward backtest..."
        docker compose exec worker python -m services.engine.backtest run "${@:2}"
        ;;

    backtest-summary)
        echo "Backtest report summary..."
        docker compose exec worker python -m services.engine.backtest summary "${@:2}"
        ;;

    backtest-compare)
        echo "Comparing backtest reports..."
        docker compose exec worker python -m services.engine.backtest compare "${@:2}"
        ;;

    seed)
        echo "Seeding leagues, seasons, and team aliases..."
        docker compose exec worker python -m services.engine.ingest seed
        ;;

    csv-backfill)
        echo "Running CSV backfill..."
        docker compose exec worker python -m services.engine.ingest csv-backfill
        ;;

    fixtures-sync)
        echo "Syncing fixtures from football-data.org..."
        docker compose exec worker python -m services.engine.ingest fixtures-sync
        ;;

    verify-ingest)
        echo "Running ingest verification..."
        docker compose exec worker python -m services.engine.ingest verify-ingest
        ;;

    aliases)
        shift
        case "${1:-review}" in
            review)
                docker compose exec worker python -m services.engine.ingest aliases review
                ;;
            confirm)
                docker compose exec worker python -m services.engine.ingest aliases confirm "${@:2}"
                ;;
            *)
                echo "Usage: ./run.sh aliases [review|confirm --source X --raw-name Y --team Z]"
                ;;
        esac
        ;;

    clean)
        echo "Stopping services and removing volumes..."
        docker compose down -v --remove-orphans
        echo "Clean complete"
        ;;

    help|*)
        echo "Football Predictor - Development Commands"
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  dev            Start all services (build + detached)"
        echo "  stop           Stop all services"
        echo "  logs           Follow service logs (optionally: ./run.sh logs api)"
        echo "  health         Show health status of API and Web"
        echo "  verify         Gate check — PASS if both services return 200"
        echo "  lint           Run linters (ruff + next lint)"
        echo "  test           Run test suites"
        echo "  test-models    Run Dixon-Coles model tests only"
        echo "  test-backtest  Run backtest test suite only"
        echo "  backtest       Run walk-forward backtest"
        echo "  backtest-summary  Print summary of a backtest JSON report"
        echo "  backtest-compare  Compare two reports for byte-identical output"
        echo "  seed           Seed leagues, seasons, and team aliases"
        echo "  csv-backfill   Backfill season CSVs from football-data.co.uk"
        echo "  fixtures-sync  Sync upcoming fixtures from football-data.org"
        echo "  verify-ingest  Run quality checks on ingested data"
        echo "  aliases        Review/confirm team aliases"
        echo "  clean          Stop services and remove volumes"
        echo "  help           Show this help message"
        ;;
esac
