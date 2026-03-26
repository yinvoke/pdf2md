#!/bin/bash
# Benchmark runner: runs all 3 tools and compares results.
#
# Usage: bash benchmark/run_all.sh [docling|marker|kreuzberg|compare|all]
# Default: all
#
# Prerequisites:
#   1. Project venv at .venv/ (for docling)
#   2. benchmark/.venv-marker/   (pip install marker-pdf)
#   3. benchmark/.venv-kreuzberg/ (pip install "kreuzberg[paddleocr]")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TARGET="${1:-all}"

run_docling() {
    echo -e "${BLUE}━━━ Running Docling benchmark ━━━${NC}"
    if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
        "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/run_docling.py"
    else
        echo -e "${YELLOW}Warning: .venv not found, using system python${NC}"
        python "$SCRIPT_DIR/run_docling.py"
    fi
    echo -e "${GREEN}✓ Docling done${NC}\n"
}

run_marker() {
    echo -e "${BLUE}━━━ Running marker benchmark ━━━${NC}"
    MARKER_VENV="$SCRIPT_DIR/.venv-marker"
    if [ ! -f "$MARKER_VENV/bin/python" ]; then
        echo -e "${RED}Error: marker venv not found at $MARKER_VENV${NC}"
        echo "Create it with:"
        echo "  python -m venv $MARKER_VENV"
        echo "  $MARKER_VENV/bin/pip install marker-pdf"
        return 1
    fi
    "$MARKER_VENV/bin/python" "$SCRIPT_DIR/run_marker.py"
    echo -e "${GREEN}✓ marker done${NC}\n"
}

run_kreuzberg() {
    echo -e "${BLUE}━━━ Running kreuzberg benchmark ━━━${NC}"
    KREUZBERG_VENV="$SCRIPT_DIR/.venv-kreuzberg"
    if [ ! -f "$KREUZBERG_VENV/bin/python" ]; then
        echo -e "${RED}Error: kreuzberg venv not found at $KREUZBERG_VENV${NC}"
        echo "Create it with:"
        echo "  python -m venv $KREUZBERG_VENV"
        echo "  $KREUZBERG_VENV/bin/pip install 'kreuzberg[paddleocr]'"
        return 1
    fi
    "$KREUZBERG_VENV/bin/python" "$SCRIPT_DIR/run_kreuzberg.py"
    echo -e "${GREEN}✓ kreuzberg done${NC}\n"
}

run_compare() {
    echo -e "${BLUE}━━━ Comparing results ━━━${NC}"
    python "$SCRIPT_DIR/compare.py"
}

case "$TARGET" in
    docling)   run_docling ;;
    marker)    run_marker ;;
    kreuzberg) run_kreuzberg ;;
    compare)   run_compare ;;
    all)
        run_docling
        run_marker
        run_kreuzberg
        run_compare
        ;;
    *)
        echo "Usage: $0 [docling|marker|kreuzberg|compare|all]"
        exit 1
        ;;
esac
