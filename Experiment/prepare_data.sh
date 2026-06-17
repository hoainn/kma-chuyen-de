#!/usr/bin/env bash
# Verify the full DongTing_Official release is unpacked into the layout train.py expects:
#   DongTing_Official/{Normal_data/*.zip, Abnormal_data/*.zip, Baseline.xlsx, syscall_64.tbl}
# The top-level Normal_data.zip / Abnormal_data.zip must be extracted (they yield the
# per-suite / per-kernel inner zips that hold the .log members).
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root: kma-chuyen-de
ROOT="DongTing_Official"

fail() { echo "FAIL: $1" >&2; exit 1; }

[ -d "$ROOT" ] || fail "$ROOT not found"
[ -f "$ROOT/Baseline.xlsx" ]  || fail "$ROOT/Baseline.xlsx missing"
[ -f "$ROOT/syscall_64.tbl" ] || fail "$ROOT/syscall_64.tbl missing"

for d in Normal_data Abnormal_data; do
  if [ ! -d "$ROOT/$d" ]; then
    if [ -f "$ROOT/$d.zip" ]; then
      echo "Extracting $ROOT/$d.zip ..."
      ( cd "$ROOT" && unzip -o -q "$d.zip" )
    else
      fail "$ROOT/$d (dir) and $ROOT/$d.zip both missing"
    fi
  fi
done

n_norm=$(ls "$ROOT"/Normal_data/*.zip 2>/dev/null | wc -l | tr -d ' ')
n_abn=$(ls "$ROOT"/Abnormal_data/*.zip 2>/dev/null | wc -l | tr -d ' ')
[ "$n_norm" -ge 1 ] || fail "no inner zips under Normal_data/"
[ "$n_abn"  -ge 1 ] || fail "no inner zips under Abnormal_data/"

echo "OK: DongTing_Official ready"
echo "  Normal_data inner zips:   $n_norm"
echo "  Abnormal_data inner zips: $n_abn"
echo "  Baseline.xlsx + syscall_64.tbl present"
