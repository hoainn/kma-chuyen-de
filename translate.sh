#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ERROR: ANTHROPIC_API_KEY environment variable is not set"
  exit 1
fi

echo "==> Running translation improvement in Docker..."

docker run --rm \
  -v "$SCRIPT_DIR:/workspace" \
  -w /workspace \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  python:3.12-slim \
  bash -c "pip install anthropic --quiet && python translate.py $*"
