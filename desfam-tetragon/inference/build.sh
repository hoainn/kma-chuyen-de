#!/usr/bin/env bash
# Build the desfam-detector container image with model artifacts baked in.
#
# Usage:
#   ./build.sh                   # builds desfam-detector:latest
#   ./build.sh v1.0              # builds desfam-detector:v1.0
#   IMG=ghcr.io/you/desfam-detector ./build.sh v1.0 push
set -euo pipefail

cd "$(dirname "$0")"

TAG="${1:-latest}"
IMG="${IMG:-desfam-detector}"
PUSH="${2:-}"

# Stage artifacts into a build context that Dockerfile can COPY from.
rm -rf artifacts
mkdir -p artifacts/outputs
# v1 artefacts (still needed for variants `ensemble` and `vae_dongting`)
cp ../outputs/if_model.joblib              artifacts/outputs/
cp ../outputs/vae_encoder.weights.h5       artifacts/outputs/
cp ../outputs/vae_decoder.weights.h5       artifacts/outputs/
cp ../outputs/model_params.json            artifacts/outputs/
cp ../outputs/scaler_fe.pkl                artifacts/outputs/
cp ../outputs/fe_report.json               artifacts/outputs/
# v2 artefacts (vae_container / lstm / cnn1d)
cp ../outputs/vae_encoder_v2_container.weights.h5  artifacts/outputs/ 2>/dev/null || true
cp ../outputs/vae_decoder_v2_container.weights.h5  artifacts/outputs/ 2>/dev/null || true
cp ../outputs/model_params_v2_container.json        artifacts/outputs/ 2>/dev/null || true
cp ../outputs/lstm_v2.weights.h5                    artifacts/outputs/ 2>/dev/null || true
cp ../outputs/cnn1d_v2.weights.h5                   artifacts/outputs/ 2>/dev/null || true
cp ../outputs/train_supervised_report.json          artifacts/outputs/ 2>/dev/null || true
cp ../../experiment/data/dongting/syscall_64.tbl artifacts/syscall_64.tbl

docker build -f Dockerfile.detector -t "${IMG}:${TAG}" .

if [[ "$PUSH" == "push" ]]; then
  docker push "${IMG}:${TAG}"
fi

rm -rf artifacts
echo "Built: ${IMG}:${TAG}"
