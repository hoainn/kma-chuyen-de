#!/usr/bin/env python3
"""
Cryptominer simulation. Generates the canonical syscall pattern of a
miner: open a TCP socket to a "pool", send a "subscribe" message, then
burn CPU on a hash loop while reading/writing the socket. We use
sha256 in tight loop rather than scrypt to keep the test container small;
the syscall trace is what matters, not the math.
"""
import hashlib
import os
import socket
import sys
import time


POOL_HOST = os.environ.get('POOL_HOST', 'bait.demo.svc.cluster.local')
POOL_PORT = int(os.environ.get('POOL_PORT', '4446'))
DURATION  = int(os.environ.get('DURATION', '15'))


def main():
    end = time.time() + DURATION
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        try:
            s.connect((POOL_HOST, POOL_PORT))
        except OSError:
            pass  # listener absent — syscall pattern is what we want
        subscribe = b'{"id":1,"method":"mining.subscribe","params":[]}\n'
        try:
            s.send(subscribe)
        except OSError:
            pass

        h = hashlib.sha256()
        ctr = 0
        while time.time() < end:
            h.update(os.urandom(64))
            ctr += 1
            if ctr % 2000 == 0:
                # Periodic "submit" packet
                try:
                    s.send(b'{"method":"mining.submit","params":["fake"]}\n')
                except OSError:
                    pass
        s.close()
    except Exception as e:
        print(f'sim error: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
