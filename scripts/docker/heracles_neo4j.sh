#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${HERACLES_COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"
SERVICE="${HERACLES_NEO4J_SERVICE:-neo4j}"

usage() {
  echo "Usage: $0 {up|down|restart|logs|ps}"
  echo
  echo "Uses Docker Compose file: ${COMPOSE_FILE}"
  echo "Neo4j defaults from Heracles compose:"
  echo "  URI:      neo4j://neo4j:7683 (preferred), neo4j://127.0.0.1:7683 (host fallback)"
  echo "  HTTP:     http://127.0.0.1:7469"
  echo "  Username: neo4j"
  echo "  Password: neo4j_pw"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not available on PATH. Install Docker before using this helper." >&2
  exit 127
fi

case "$1" in
  up)
    docker compose -f "${COMPOSE_FILE}" up -d "${SERVICE}"
    ;;
  down)
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  restart)
    docker compose -f "${COMPOSE_FILE}" restart "${SERVICE}"
    ;;
  logs)
    docker compose -f "${COMPOSE_FILE}" logs -f "${SERVICE}"
    ;;
  ps)
    docker compose -f "${COMPOSE_FILE}" ps
    ;;
  *)
    usage
    exit 2
    ;;
esac
