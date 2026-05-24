#!/usr/bin/env bash
# Remove all DeSFAM research workloads from the demo namespace.
DIR="$(cd "$(dirname "$0")" && pwd)"
kubectl delete -f "$DIR/normal/" --ignore-not-found
kubectl delete -f "$DIR/attack/" --ignore-not-found
echo "Done."
