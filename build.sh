#!/bin/bash
# Usage: bash build.sh [en|vi|eda]
#   en  → build English extraction (main_en.tex)
#   vi  → build Vietnamese translation (main.tex, default)
#   eda → build our DongTing EDA + modelling report (main_eda.tex)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LATEX_DIR="$SCRIPT_DIR/latex"

TARGET="${1:-vi}"
if [ "$TARGET" = "en" ]; then
  TEXFILE="main_en.tex"
  JOBNAME="main_en"
  OUTPUT_DIR="$SCRIPT_DIR/output_en"
  BIBFILE="references.bib"
elif [ "$TARGET" = "eda" ]; then
  TEXFILE="main_eda.tex"
  JOBNAME="main_eda"
  OUTPUT_DIR="$SCRIPT_DIR/output_eda"
  BIBFILE="references_eda.bib"
else
  TEXFILE="main.tex"
  JOBNAME="main"
  OUTPUT_DIR="$SCRIPT_DIR/output"
  BIBFILE="references.bib"
fi

mkdir -p "$OUTPUT_DIR"

IMAGE="texlive/texlive:latest"

echo "==> Pulling LaTeX image (if needed)..."
docker pull "$IMAGE" --quiet

echo "==> Building: $TEXFILE  →  $OUTPUT_DIR/${JOBNAME}.pdf"

echo "==> Running lualatex (pass 1)..."
docker run --rm \
  -v "$LATEX_DIR:/workspace" \
  -v "$OUTPUT_DIR:/output" \
  -w /workspace \
  "$IMAGE" \
  lualatex -interaction=nonstopmode -jobname="$JOBNAME" -output-directory=/output "$TEXFILE" || true

echo "==> Running bibtex..."
cp "$LATEX_DIR/$BIBFILE" "$OUTPUT_DIR/"
docker run --rm \
  -v "$LATEX_DIR:/workspace" \
  -v "$OUTPUT_DIR:/output" \
  -w /output \
  "$IMAGE" \
  bibtex "$JOBNAME" || true

echo "==> Running lualatex (pass 2)..."
docker run --rm \
  -v "$LATEX_DIR:/workspace" \
  -v "$OUTPUT_DIR:/output" \
  -w /workspace \
  "$IMAGE" \
  lualatex -interaction=nonstopmode -jobname="$JOBNAME" -output-directory=/output "$TEXFILE"

echo "==> Running lualatex (pass 3 - final)..."
docker run --rm \
  -v "$LATEX_DIR:/workspace" \
  -v "$OUTPUT_DIR:/output" \
  -w /workspace \
  "$IMAGE" \
  lualatex -interaction=nonstopmode -jobname="$JOBNAME" -output-directory=/output "$TEXFILE"

echo ""
echo "==> Build complete: $OUTPUT_DIR/${JOBNAME}.pdf"
