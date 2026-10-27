"""
Serve a p4 store's .pt ranker on localhost, the checkpoint loaded once and reloaded the moment a new one lands.

The vlds plugin's hooks run a fresh process per call, too short-lived to import torch and load a checkpoint each
time; this server holds models/ranker.pt in memory instead and answers in milliseconds. It watches the file's
modification time between requests, so a background retrain that swaps in a new checkpoint is picked up at the next
request, with no restart: the model keeps running while it trains.

    python -m phi_prune.p4serve --p4 <p4 root> [--port 0] [--idle-hours 4]

It writes <p4>/models/ranker.server.json — {"port", "pid", "round"} — when it is ready, and removes it when it stops
(after --idle-hours without a request). Endpoints, JSON in and out, bound to 127.0.0.1 only:

    GET  /health                          → {"ok": true, "round": N, "threshold": t}
    POST /score {"task", "texts": [...]}   → {"scores": [...], "round": N, "threshold": t, "per_file": 6}
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import torch

from phi_prune.p4rank import load, models_dir, write_text_atomic

CACHE_MAX = 50_000            # entry embeddings kept between requests, by text


class Holder:
    """The loaded ranker, swapped whole when the checkpoint changes on disk."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.model = None
        self.meta: Dict = {}
        self.mtime = 0.0
        self.cache: Dict[str, torch.Tensor] = {}
        self.last = time.time()

    def current(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return self.model, self.meta
        if m != self.mtime:
            with self.lock:
                if m != self.mtime:
                    try:
                        model, meta = load(self.path)
                    except Exception:  # noqa: BLE001 — a checkpoint mid-swap or broken: keep serving the old one
                        return self.model, self.meta
                    self.model, self.meta, self.mtime, self.cache = model, meta, m, {}
        return self.model, self.meta

    @torch.no_grad()
    def score(self, task: str, texts):
        model, meta = self.current()
        if model is None:
            return None, meta
        q = model.encode([task], "task")[0]
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        for s in range(0, len(missing), 512):
            idx = missing[s:s + 512]
            emb = model.encode([texts[i] for i in idx], "entry")
            for i, e in zip(idx, emb):
                self.cache[keys[i]] = e
        if len(self.cache) > CACHE_MAX:
            self.cache = {k: self.cache[k] for k in keys if k in self.cache}
        return [float(self.cache[k] @ q) for k in keys], meta


def make_handler(holder: Holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the hooks read the answers, not a log
            pass

        def reply(self, code: int, body: Dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            holder.last = time.time()
            if self.path != "/health":
                return self.reply(404, {"error": "not found"})
            model, meta = holder.current()
            self.reply(200, {"ok": model is not None, "round": meta.get("round"), "threshold": meta.get("threshold")})

        def do_POST(self):
            holder.last = time.time()
            if self.path != "/score":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                scores, meta = holder.score(str(req["task"]), [str(t) for t in req["texts"]])
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": f"{type(e).__name__}: {e}"})
            if scores is None:
                return self.reply(503, {"error": "no checkpoint loaded"})
            self.reply(200, {"scores": scores, "round": meta.get("round"), "threshold": meta.get("threshold"),
                             "per_file": meta.get("per_file", 6)})

    return Handler


def write_atomic(path: str, text: str) -> None:
    write_text_atomic(text, path)


def serve(p4: str, port: int = 0, idle_hours: float = 4.0, ready: Optional[threading.Event] = None):
    """Run until idle; returns the server (tests stop it with .shutdown())."""
    mdir = models_dir(p4)
    holder = Holder(os.path.join(mdir, "ranker.pt"))
    _model, meta = holder.current()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(holder))
    info = os.path.join(mdir, "ranker.server.json")
    os.makedirs(mdir, exist_ok=True)
    write_atomic(info, json.dumps({"port": server.server_address[1], "pid": os.getpid(), "round": meta.get("round")}))

    def idle_watch():
        while True:
            time.sleep(30)
            if time.time() - holder.last > idle_hours * 3600:
                server.shutdown()
                return
    threading.Thread(target=idle_watch, daemon=True).start()
    if ready is not None:
        ready.set()
    try:
        server.serve_forever()
    finally:
        try:
            with open(info, encoding="utf-8") as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(info)
        except (OSError, ValueError):
            pass
    return server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phi_prune.p4serve")
    ap.add_argument("--p4", required=True)
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port; the chosen one is in ranker.server.json")
    ap.add_argument("--idle-hours", type=float, default=4.0)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    serve(args.p4, args.port, args.idle_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
