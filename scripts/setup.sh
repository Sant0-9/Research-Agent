#!/usr/bin/env bash
# Research Agent - Setup Script
# Run this once to set up the development environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Research Agent Setup ==="
echo "Project root: $PROJECT_ROOT"

# Check Python version
echo ""
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

if [[ "$(echo "$PYTHON_VERSION < 3.12" | bc -l)" -eq 1 ]]; then
    echo "ERROR: Python 3.12 or higher is required."
    exit 1
fi

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up virtual environment..."
if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    python3 -m venv "$PROJECT_ROOT/.venv"
    echo "Created virtual environment at .venv"
else
    echo "Virtual environment already exists at .venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -e "$PROJECT_ROOT[dev]"

# Create .env from .env.example if it doesn't exist
echo ""
echo "Checking environment configuration..."
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "Created .env from .env.example"
        echo "IMPORTANT: Edit .env and add your API keys before running the application."
    else
        echo "WARNING: .env.example not found. Create .env manually."
    fi
else
    echo ".env already exists"
fi

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/storage"

# Check Docker
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo "Docker is installed: $(docker --version)"
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        echo "Docker Compose is available"
    else
        echo "WARNING: Docker Compose not found. Required for running services."
    fi
else
    echo "WARNING: Docker not installed. Required for running services."
fi

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (OPENAI_API_KEY, TAVILY_API_KEY, etc.)"
echo "  2. Activate the virtual environment: source .venv/bin/activate"
echo "  3. Start services: ./scripts/run_local.sh"
echo "  4. Run tests: ./scripts/test.sh"
echo ""
