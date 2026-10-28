"""
Serve a p4 store's .pt ranker on localhost, the checkpoint loaded once and reloaded the moment a new one lands.

The vlds plugin's hooks run a fresh process per call, too short-lived to import torch and load a checkpoint each
time; this server holds models/ranker.pt in memory instead and answers in milliseconds. It watches the file's
modification times between requests (ranker.pt, the sidecar ranker.json, the anchor), so a background retrain that
ships a round, or a pull that brings one, is picked up at the next request with no restart: the model keeps running
while it trains. What it loads is phi_rank.load_current: the chain's build, rebuilt from the anchor and its φ-deltas
when ranker.pt is not a build of the chain the sidecar names.

    python -m phi_serve --p4 <p4 root> [--port 0] [--idle-hours 4]

It writes <p4>/models/ranker.server.json — {"port", "pid", "round"} — when it is ready, and removes it when it stops
(after --idle-hours without a request). Endpoints, JSON in and out, bound to 127.0.0.1 only:

    GET  /health                          → {"ok": true, "round": N, "threshold": t}
    POST /score {"task", "texts": [...], "effort"?}
                                          → {"scores": [...], "round": N, "threshold": t, "per_file": 6, "source"}

"effort" picks the answer: "medium" is the one node of the store's node tree (phi_rank.tree) its pruning mask routes
the task to, "high" every node of the tree averaged; anything else, or no tree in models/, is the single ranker.
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

import phi_rank.tree as tree_mod
from phi_rank import ANCHOR, CURRENT, SIDECAR, load_current, models_dir, write_text_atomic

CACHE_MAX = 50_000            # entry embeddings kept between requests, by text
MAX_BODY = 16 << 20           # a /score request larger than this is refused (413), never read


class Holder:
    """The loaded ranker, swapped whole when the checkpoint changes on disk."""

    def __init__(self, p4: str):
        self.p4 = os.path.realpath(p4)      # named in every reply: a client trusts only its own store's server
        self.watch = [os.path.join(models_dir(p4), n) for n in (CURRENT, SIDECAR, ANCHOR)]
        self.lock = threading.Lock()
        self.model = None
        self.meta: Dict = {}
        self.sig = None
        self.cache: Dict[str, torch.Tensor] = {}
        self.last = time.time()
        self.tree_path = os.path.join(models_dir(p4), tree_mod.TREE_JSON)
        self.tree = None
        self.tree_sig = None

    def current_tree(self):
        """The node tree the effort tiers answer from, reloaded when tree.json changes; None when there is none."""
        try:
            sig = os.path.getmtime(self.tree_path)
        except OSError:
            return None
        if sig != self.tree_sig:
            with self.lock:
                if sig != self.tree_sig:
                    try:
                        self.tree = tree_mod.Tree(self.p4)
                    except Exception:  # noqa: BLE001 — a tree mid-write or broken: keep the old one
                        return self.tree
                    self.tree_sig = sig
        return self.tree

    def signature(self):
        out = []
        for p in self.watch:
            try:
                out.append(os.path.getmtime(p))
            except OSError:
                out.append(None)
        return tuple(out)

    def current(self):
        sig = self.signature()
        if sig != self.sig:
            with self.lock:
                if sig != self.sig:
                    try:
                        model, meta = load_current(self.p4)
                    except Exception:  # noqa: BLE001 — a checkpoint mid-swap or broken: keep serving the old one
                        return self.model, self.meta
                    # the build load_current may have just written moved ranker.pt's mtime: take the signature now
                    self.model, self.meta, self.sig, self.cache = model, meta, self.signature(), {}
        return self.model, self.meta

    @torch.no_grad()
    def score(self, task: str, texts):
        """Scores against the model current at the call, its entry embeddings cached by text. The cache is used
        under the holder's lock and only for the model it was built with, so concurrent requests and a reload in
        between never read another model's embeddings or a cache being trimmed."""
        model, meta = self.current()
        if model is None:
            return None, meta
        q = model.encode([task], "task")[0]
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        with self.lock:
            cache = self.cache if self.model is model else {}
            have = {k: cache[k] for k in keys if k in cache}
        missing = [i for i, k in enumerate(keys) if k not in have]
        for s in range(0, len(missing), 512):
            idx = missing[s:s + 512]
            for i, e in zip(idx, model.encode([texts[i] for i in idx], "entry")):
                have[keys[i]] = e
        with self.lock:
            if self.model is model:
                self.cache.update(have)
                if len(self.cache) > CACHE_MAX:
                    self.cache = dict(have)
        return [float(have[k] @ q) for k in keys], meta


def make_handler(holder: Holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the hooks read the answers, not a log
            pass

        def reply(self, code: int, body: Dict):
            data = json.dumps({**body, "p4": holder.p4}).encode("utf-8")
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
                size = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self.reply(400, {"error": "a Content-Length is required"})
            if size <= 0 or size > MAX_BODY:
                return self.reply(413 if size > MAX_BODY else 400, {"error": f"body of {size} bytes refused"})
            try:
                req = json.loads(self.rfile.read(size))
                task, texts = str(req["task"]), [str(t) for t in req["texts"]]
                effort = str(req.get("effort") or "")
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": f"{type(e).__name__}: {e}"})
            try:
                tree = holder.current_tree() if effort in tree_mod.EFFORTS else None
                if tree is not None:
                    # the effort tiers: medium = the node the pruning matrix routes to, high = every node
                    scores, source = tree.score(task, texts, effort)
                    return self.reply(200, {"scores": scores, "round": tree.meta.get("trained"), "threshold": 0.0,
                                            "per_file": 6, "source": source})
                scores, meta = holder.score(task, texts)
            except Exception as e:  # noqa: BLE001 — the model failed: say so, and keep serving
                return self.reply(500, {"error": f"{type(e).__name__}: {e}"})
            if scores is None:
                return self.reply(503, {"error": "no checkpoint loaded"})
            self.reply(200, {"scores": scores, "round": meta.get("round"), "threshold": meta.get("threshold"),
                             "per_file": meta.get("per_file", 6), "source": "ranker"})

    return Handler


def write_atomic(path: str, text: str) -> None:
    write_text_atomic(text, path)


def serve(p4: str, port: int = 0, idle_hours: float = 4.0, ready: Optional[threading.Event] = None,
          info_name: str = "ranker.server.json"):
    """Run until idle. The port file is ranker.server.json, the one the plugin's hooks read; a check run beside a live
    server passes another info_name so it never takes that server's place."""
    mdir = models_dir(p4)
    holder = Holder(p4)
    _model, meta = holder.current()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(holder))
    info = os.path.join(mdir, info_name)
    os.makedirs(mdir, exist_ok=True)
    write_atomic(info, json.dumps({"port": server.server_address[1], "pid": os.getpid(), "round": meta.get("round"),
                                   "p4": holder.p4}))
    try:
        os.remove(os.path.join(mdir, ".server-start"))   # the plugin's start lock: this start has landed
    except OSError:
        pass

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
    ap = argparse.ArgumentParser(prog="python -m phi_serve")
    ap.add_argument("--p4", required=True)
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port; the chosen one is in ranker.server.json")
    ap.add_argument("--idle-hours", type=float, default=4.0)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    serve(args.p4, args.port, args.idle_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
