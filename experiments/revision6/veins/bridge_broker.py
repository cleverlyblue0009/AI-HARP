"""Revision 6: TCP broker inside WSL between the Veins app and the decision server on Windows (stdlib only).

    python3 bridge_broker.py <app_port> <server_port>

WSL (NAT mode) cannot open connections to Windows here (the Windows firewall drops them), but Windows reaches WSL's
listening ports on localhost. So the Windows decision server dials in on <server_port> and waits; the app connects on
127.0.0.1:<app_port>; the broker pairs one server connection with one app connection (one replay) and copies bytes
both ways unchanged until either side closes, then closes both and waits for the next pair. It computes nothing.
"""

from __future__ import annotations

import select
import socket
import sys


def listener(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s


def pipe(a: socket.socket, b: socket.socket) -> None:
    for s in (a, b):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    peer = {a: b, b: a}
    while True:
        r, _, _ = select.select([a, b], [], [])
        for s in r:
            data = s.recv(65536)
            if not data:
                return
            peer[s].sendall(data)


def main() -> int:
    app_port, srv_port = int(sys.argv[1]), int(sys.argv[2])
    la, ls = listener("127.0.0.1", app_port), listener("0.0.0.0", srv_port)
    print(f"[broker] app 127.0.0.1:{app_port} <-> server :{srv_port}", flush=True)
    while True:
        srv, _ = ls.accept()                     # the decision server, ready for the next replay
        app, _ = la.accept()                     # the replay's app
        try:
            pipe(app, srv)
        finally:
            app.close()
            srv.close()


if __name__ == "__main__":
    raise SystemExit(main())
