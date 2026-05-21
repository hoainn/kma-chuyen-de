#!/usr/bin/env bash
set -e

HOST=mongo
DB=syzbot_DB
DATADIR=/dongting-db
TMPDIR=/tmp/dt_import

mkdir -p "$TMPDIR"

# Install unzip (mongo:6 image is Debian-based but ships without it)
if ! command -v unzip &>/dev/null; then
  echo "[init] Installing unzip..."
  apt-get update -qq && apt-get install -y -qq unzip > /dev/null
fi

echo "[init] Waiting for MongoDB at $HOST..."
until mongosh --host "$HOST" --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  sleep 3
done
echo "[init] MongoDB ready."

# ── helper: import one BSON if collection is empty ─────────────────────────
import_if_empty() {
  local zip="$1" bson="$2" col="$3"
  local count
  count=$(mongosh --host "$HOST" "$DB" --quiet \
    --eval "db['$col'].estimatedDocumentCount()" 2>/dev/null || echo 0)

  if [ "$count" -gt 0 ] 2>/dev/null; then
    echo "[init] $col already has $count docs — skipping."
    return
  fi

  echo "[init] Extracting $zip ..."
  unzip -o "$DATADIR/$zip" -d "$TMPDIR" > /dev/null

  echo "[init] Importing $bson into $col ..."
  mongorestore \
    --host "$HOST" \
    --db "$DB" \
    --collection "$col" \
    --drop \
    "$TMPDIR/$bson"

  echo "[init] $col imported."
  rm -f "$TMPDIR/$bson"
}

# Baseline metadata (small, ~6 MB unzipped — always import)
import_if_empty \
  kernel_convert_baseline.zip \
  kernel_convert_baseline.bson \
  kernel_convert_baseline

# Normal syscall sequences (~510 MB unzipped)
import_if_empty \
  kernel_syscall_normal_strace.zip \
  kernel_syscall_normal_strace.bson \
  kernel_syscall_normal_strace

# Attack syscall sequences (~3 GB unzipped — takes several minutes)
import_if_empty \
  kernel_syscallhook_bugpoc_trace_sum.zip \
  kernel_syscallhook_bugpoc_trace_sum.bson \
  kernel_syscallhook_bugpoc_trace_sum

echo "[init] All collections ready in $DB."
