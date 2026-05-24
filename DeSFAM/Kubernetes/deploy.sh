#!/usr/bin/env bash
# Deploy all DeSFAM research workloads to the demo namespace.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Normal workloads ==="
kubectl apply -f "$DIR/normal/"

echo ""
echo "=== Attack workloads ==="
kubectl apply -f "$DIR/attack/"

echo ""
echo "=== Pod status ==="
kubectl get pods -n demo -o wide
