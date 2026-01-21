#!/usr/bin/env bash
# Research Agent - Test Runner
# Runs all tests with coverage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
COVERAGE=false
VERBOSE=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cov|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -k)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cov, --coverage  Run with coverage report"
            echo "  -v, --verbose      Verbose output"
            echo "  -k PATTERN         Run tests matching pattern"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo "WARNING: Virtual environment not found. Run ./scripts/setup.sh first."
    fi
fi

echo "=== Running Research Agent Tests ==="
echo ""

# Build pytest command
PYTEST_CMD="pytest"

if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing --cov-report=html"
fi

if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
fi

if [[ -n "$FILTER" ]]; then
    PYTEST_CMD="$PYTEST_CMD -k \"$FILTER\""
fi

# Run tests
echo "Running: $PYTEST_CMD"
echo ""
eval $PYTEST_CMD

# Show coverage report location if generated
if [[ "$COVERAGE" == true ]]; then
    echo ""
    echo "Coverage report generated at: htmlcov/index.html"
fi
