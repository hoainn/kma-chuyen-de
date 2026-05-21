#!/usr/bin/env bash
# Build the experience report PDFs (English + Vietnamese).
#
# Usage:
#   bash build.sh             # builds both main_en.pdf and main_vi.pdf
#   bash build.sh en          # builds main_en.pdf only
#   bash build.sh vi          # builds main_vi.pdf only
#
# Uses the texlive/texlive container (LuaLaTeX + IEEEtran + babel). If
# you have a local TeX Live install, prefer:
#   latexmk -lualatex -interaction=nonstopmode main_en.tex
#   latexmk -lualatex -interaction=nonstopmode main_vi.tex
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-texlive/texlive:latest}"
TARGET="${1:-all}"

build_one() {
  local doc="$1"
  echo "==> Building ${doc}.pdf"
  # Explicit multi-pass: lualatex → bibtex → lualatex → lualatex
  # Ensures \cite{} and \ref{} both resolve before the final PDF is written.
  docker run --rm -v "$PWD:/work" -w /work "$IMAGE" bash -c "
    set -e
    lualatex -interaction=nonstopmode '${doc}.tex' > '${doc}_build.log' 2>&1
    bibtex '${doc}'                                >> '${doc}_build.log' 2>&1
    lualatex -interaction=nonstopmode '${doc}.tex' >> '${doc}_build.log' 2>&1
    lualatex -interaction=nonstopmode '${doc}.tex' >> '${doc}_build.log' 2>&1
    if grep -qE 'Citation.*undefined|Reference.*undefined' '${doc}.log'; then
      echo 'FAIL: undefined citations or references — see ${doc}.log' >&2
      exit 1
    fi
  "
  echo "    Output: $PWD/${doc}.pdf"
}

case "$TARGET" in
  en)  build_one main_en ;;
  vi)  build_one main_vi ;;
  all) build_one main_en; build_one main_vi ;;
  *)   echo "Usage: $0 [en|vi|all]"; exit 1 ;;
esac

# Tidy auxiliary files but keep the PDFs.
docker run --rm -v "$PWD:/work" -w /work "$IMAGE" latexmk -c >/dev/null 2>&1 || true
