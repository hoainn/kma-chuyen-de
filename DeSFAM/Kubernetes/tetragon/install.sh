#!/usr/bin/env bash
# Install Tetragon via Helm and apply the DeSFAM TracingPolicy.
# Tested on: K3s / RKE2 / kubeadm, kernel >= 5.10
set -e

NAMESPACE="kube-system"
RELEASE="tetragon"
POLICY="$(cd "$(dirname "$0")" && pwd)/tracing-policy.yaml"

echo "==> Adding Cilium Helm repo..."
helm repo add cilium https://helm.cilium.io
helm repo update

echo "==> Installing Tetragon (namespace: $NAMESPACE)..."
helm upgrade --install "$RELEASE" cilium/tetragon \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --set tetragon.grpc.address="0.0.0.0:54321" \
  --set tetragon.exportFilename="/var/run/cilium/tetragon/tetragon.log" \
  --wait

echo "==> Waiting for Tetragon pods to be ready..."
kubectl rollout status daemonset/tetragon -n "$NAMESPACE" --timeout=120s

echo "==> Applying DeSFAM TracingPolicy..."
kubectl apply -f "$POLICY"

echo ""
echo "==> Tetragon gRPC is available on each node at port 54321."
echo "    Expose via NodePort or LoadBalancer — see NodePort manifest below:"
echo ""
echo "    kubectl apply -f nodeport.yaml"
echo ""
echo "==> Verify: kubectl get tracingpolicy"
