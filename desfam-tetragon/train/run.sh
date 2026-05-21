#!/usr/bin/env bash
# Start the DongTing experiment environment.
# Usage:
#   bash run.sh          → start all services
#   bash run.sh down     → stop and remove containers
#   bash run.sh logs     → stream logs
set -e

cd "$(dirname "$0")"

case "${1:-up}" in
  up)
    echo "Starting containers..."
    docker compose up --build -d
    echo ""
    echo "JupyterLab: http://localhost:8888"
    echo "  (wait ~30 s for mongo-init to finish importing data)"
    echo ""
    echo "To follow logs: bash run.sh logs"
    ;;
  down)
    docker compose down
    ;;
  logs)
    docker compose logs -f
    ;;
  *)
    echo "Usage: bash run.sh [up|down|logs]"
    ;;
esac
