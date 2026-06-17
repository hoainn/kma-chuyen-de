#!/usr/bin/env bash
# Full-dataset SyscallAD reproduction + rigorous evaluation, all in Docker.
#   1) verify dataset layout
#   2) train.py (proven engine) on the full DongTing release -> model/ + results.json
#   3) rigorous_eval.py -> rigorous_metrics.json + SUMMARY.md
# Outputs land in Experiment/results/run_<timestamp>/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"                       # kma-chuyen-de
DATA="$ROOT/DongTing_Official"
TRAIN_DIR="$ROOT/DeSFAM/training"
IMAGE="desfam-experiment:latest"

TS="$(date +%Y%m%d-%H%M%S)"
RUN="$HERE/results/run_$TS"
MODEL="$RUN/model"
mkdir -p "$MODEL"

echo "==> [1/4] verify dataset"
bash "$HERE/prepare_data.sh"

echo "==> [2/4] build training image"
docker build -t "$IMAGE" "$TRAIN_DIR"

COMMON_MOUNTS=(
  -v "$DATA":/data/dongting:ro
  -v "$MODEL":/model
  -v "$TRAIN_DIR":/workspace
  -v "$HERE":/experiment
  --env-file "$HERE/config.env"
  -w /workspace
)

echo "==> [3/4] train on FULL dataset (this is the long step)"
docker run --rm "${COMMON_MOUNTS[@]}" "$IMAGE" python train.py 2>&1 | tee "$RUN/train.log"

echo "==> [4/4] rigorous evaluation"
docker run --rm "${COMMON_MOUNTS[@]}" -e PYTHONPATH=/workspace -e TRAIN_DIR=/workspace \
  "$IMAGE" python /experiment/rigorous_eval.py 2>&1 | tee "$RUN/rigorous_eval.log"

# Surface the headline artifacts at the run root.
cp -f "$MODEL/results.json"          "$RUN/results.json"          2>/dev/null || true
cp -f "$MODEL/rigorous_metrics.json" "$RUN/rigorous_metrics.json" 2>/dev/null || true
cp -f "$MODEL/SUMMARY.md"            "$RUN/SUMMARY.md"            2>/dev/null || true

echo
echo "==> DONE. Results in: $RUN"
echo "    - results.json           (window-level, report §7 Table I)"
echo "    - rigorous_metrics.json  (sequence-level, ablation, parity, success)"
echo "    - SUMMARY.md             (human-readable; maps onto report §7/§8)"
[ -f "$RUN/SUMMARY.md" ] && { echo; echo "----- SUMMARY.md -----"; cat "$RUN/SUMMARY.md"; }
