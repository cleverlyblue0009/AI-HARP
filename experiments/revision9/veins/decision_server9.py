"""Revision 9: decision server = the revision-8 one (experiments/revision8/veins/decision_server8.py, unchanged), with
STRUCT parameters read from the run directory named in the replay configuration:

    {"policy": "struct", "run": "plain_s0", "train_dir": "results/revision9/train", "delta": 1.0, ...}

`train_dir` defaults to results/revision8/train (revision-8 behaviour). AI-HARP requests (gate check 2) go through the
unchanged revision-6 Session. Request format, streams, replies and logs are those of revision 8.

    python -m experiments.revision9.veins.decision_server9 --connect 127.0.0.1:5801 --config <worker cfg.json>
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
from pathlib import Path

import numpy as np

from common.config import PROJECT_ROOT
from experiments.revision6.veins import decision_server as ds6
from experiments.revision8.veins import decision_server8 as ds8


def struct_policy(cfg: dict):
    from agents.structured import StructuredPolicy
    tdir = PROJECT_ROOT / cfg.get("train_dir", "results/revision8/train")
    theta = json.loads((tdir / cfg["run"] / "theta.json").read_text())["theta"]
    pol = StructuredPolicy(theta=theta, delta=float(cfg["delta"]))
    pol.reset(int(cfg["n_vehicles"]), np.random.default_rng([int(cfg["seed"]), ds6.STREAM_SALT, 1]))
    return pol


class Session9(ds8.Session8):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kind = cfg.get("policy", "ai_harp")
        self.k: dict[tuple[int, int], int] = {}
        self.inner = None
        if self.kind == "struct":
            self.policy = struct_policy(cfg)
        else:
            self.inner = ds6.Session(cfg)
        self.geo = (ds8.Geometric(cfg["scenario"], cfg["density"], cfg["seed"])
                    if cfg.get("neighbour_table") == "geometric" else None)


def session(conn: socket.socket, cfg_path: Path) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf, n, sess, log, cfg = b"", 0, None, None, {}
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if sess is None:
                    cfg = json.loads(cfg_path.read_text())
                    sess = Session9(cfg)
                    log = open(cfg["log"], "a", encoding="utf-8")
                    log.write(json.dumps({"session": cfg}) + "\n")
                q = json.loads(line)
                reply, info = sess.decide(q)
                conn.sendall(reply.encode() + b"\n")
                log.write(json.dumps({"req": q, "reply": reply, **info}) + "\n")
                n += 1
    finally:
        if log:
            log.close()
        conn.close()


def trace(path: Path) -> dict:
    """Offline re-send of a logged replay to a fresh Session9; every reply must be reproduced."""
    op = gzip.open if path.suffix == ".gz" else open
    lines = [json.loads(l) for l in op(path, "rt", encoding="utf-8")]
    s = Session9(next(x["session"] for x in lines if "session" in x))
    n = mism = 0
    for x in lines:
        if "req" in x:
            reply, _ = s.decide(x["req"])
            n += 1
            mism += reply != x["reply"]
    return {"log": path.name, "requests": n, "mismatches": mism, "pass": bool(n > 0 and mism == 0)}


def main(argv=None) -> int:
    import time
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--connect", required=True)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args(argv)
    print(f"[env] {sys.executable} numpy {np.__version__}", flush=True)
    host, port = a.connect.rsplit(":", 1)
    while True:
        try:
            conn = socket.create_connection((host, int(port)))
        except OSError:
            time.sleep(1.0)
            continue
        session(conn, a.config)


if __name__ == "__main__":
    raise SystemExit(main())
