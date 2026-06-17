#!/usr/bin/env bash
# Biên dịch báo cáo bằng XeLaTeX qua Docker (không cài LaTeX trên máy chủ).
# Engine XeLaTeX cần cho fontspec (tiếng Việt). Dùng image texlive/texlive.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="texlive/texlive:latest"
echo "==> Pull image (neu can)..."
docker pull "$IMAGE" --quiet >/dev/null

run() { docker run --rm -v "$PWD:/w" -w /w "$IMAGE" "$@"; }

echo "==> XeLaTeX pass 1..."
run xelatex -interaction=nonstopmode -halt-on-error main.tex || true
echo "==> BibTeX..."
run bibtex main || true
echo "==> XeLaTeX pass 2..."
run xelatex -interaction=nonstopmode -halt-on-error main.tex || true
echo "==> XeLaTeX pass 3 (final)..."
run xelatex -interaction=nonstopmode -halt-on-error main.tex

echo "==> Built: $(pwd)/main.pdf"
