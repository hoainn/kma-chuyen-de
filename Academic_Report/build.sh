#!/usr/bin/env bash
# Build the report. Primary path = Tectonic (what claude-prism bundles).
# Fallbacks: local xelatex, or the texlive Docker image. Output: main.pdf
set -e
cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  echo "==> tectonic"
  tectonic -X compile main.tex   # auto-fetches IEEEtran.cls + IEEEtran.bst
elif command -v xelatex >/dev/null 2>&1 && kpsewhich IEEEtran.cls >/dev/null 2>&1; then
  echo "==> local xelatex + bibtex"
  xelatex -interaction=nonstopmode main.tex
  bibtex main || true
  xelatex -interaction=nonstopmode main.tex
  xelatex -interaction=nonstopmode main.tex
else
  echo "==> docker texlive/texlive (xelatex + bibtex)"
  docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest bash -c \
    "xelatex -interaction=nonstopmode main.tex && bibtex main && \
     xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex"
fi
echo "==> done: main.pdf"
