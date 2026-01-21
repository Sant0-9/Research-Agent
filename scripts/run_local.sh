#!/usr/bin/env bash
# Research Agent - Local Development Runner
# Starts all services for local development

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
GPU_MODE=false
DETACHED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_MODE=true
            shift
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu        Start with local GPU brain service"
            echo "  -d, --detached  Run in detached mode"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if .env exists
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "ERROR: .env file not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Build compose command
COMPOSE_CMD="docker compose -f docker-compose.dev.yml"

if [[ "$GPU_MODE" == true ]]; then
    echo "Starting with GPU brain service..."
    COMPOSE_CMD="docker compose -f docker-compose.yml --profile gpu"
fi

# Add detached flag if requested
if [[ "$DETACHED" == true ]]; then
    COMPOSE_CMD="$COMPOSE_CMD up -d"
else
    COMPOSE_CMD="$COMPOSE_CMD up"
fi

echo "=== Starting Research Agent (Development) ==="
echo ""
echo "Services:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Redis: localhost:6379"
echo "  - PostgreSQL: localhost:5432"
if [[ "$GPU_MODE" == true ]]; then
    echo "  - Brain (vLLM): http://localhost:8001"
fi
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run docker compose
$COMPOSE_CMD
