"""
The nested p4 store as a worked example of phi over binary: the fixture and every measurement in
.docs/p4/nested-store.md.

A p4 session started with a store as its working directory grows a store inside the store
(<dir>/.claude/p4/.claude/p4/...), because the vlds hooks put a store at <working dir>/.claude/p4.
This script grows that nesting the way it happened: at each level a session is started (the vlds and
src-fragger SessionStart hooks, a prompt, the turn's close) with the previous level's store as its
working directory. Nothing is copied. Then it measures:

  1. every store node's path as digits (p4 = 1, .claude = 0): does `11` ever occur, and what else holds
  2. a stream of node paths parsed back with no length headers by phi_prune.gaps.read_codes
  3. the same stream in binary: what a fixed width or a length field costs to recover it, in bits
  4. each node's lvalue (its nested .claude/ subtree) against its rvalue (its own content), by depth

Run (2026-10-27):
  python scripts/session/nested_store.py --plugins <p4 checkout>/lbiza/plugins --depth 8 [--sessions-outer 1]
It prints the tables the note quotes and writes them as JSON beside the fixture.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phi_prune.gaps import fib_code, read_codes  # noqa: E402


# ─── the fixture: one session per level, each started in the previous level's store ─────────────────────

def run_hook(plugin_root, sub, payload, project):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=project, CLAUDE_PLUGIN_ROOT=plugin_root)
    subprocess.run(["sh", os.path.join(plugin_root, "hooks", "run-hook.sh" if sub[0] != "@" else "session-open.sh"),
                    *(sub[1:].split() if sub[0] == "@" else sub.split())],
                   input=json.dumps(payload).encode(), env=env, cwd=project,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def start_session(plugins, project, prompt):
    """What a session leaves in <project>/.claude/p4: SessionStart (vlds, src-fragger), one prompt, the close."""
    sid = str(uuid.uuid4())
    transcript = os.path.join(tempfile.gettempdir(), f"nested-{sid}.jsonl")
    open(transcript, "w").close()
    base = {"session_id": sid, "cwd": project, "transcript_path": transcript}
    vlds, frag = os.path.join(plugins, "vlds"), os.path.join(plugins, "src-fragger")
    start = {**base, "hook_event_name": "SessionStart", "source": "startup"}
    run_hook(vlds, "session-open", start, project)
    for sub in ("@seed", "register-lint", "notebook-out"):
        run_hook(frag, sub, start, project)
    run_hook(vlds, "prompt-open", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}, project)
    run_hook(vlds, "turn-close", {**base, "hook_event_name": "Stop"}, project)
    os.remove(transcript)
    return sid


def grow(plugins, root, depth, sessions_outer):
    """Level 1's store is <root>/.claude/p4; level k+1 is started inside level k's store."""
    project, stores = root, []
    for level in range(1, depth + 1):
        n = sessions_outer if level < depth else 1
        for i in range(n):
            start_session(plugins, project, f"work at nesting level {level}, session {i + 1}")
        store = os.path.join(project, ".claude", "p4")
        stores.append(store)
        project = store
    return stores


# ─── the measurements ───────────────────────────────────────────────────────────────────────────────

def node_digits(rel_parts):
    return "".join("1" if p == "p4" else "0" if p == ".claude" else "?" for p in rel_parts)


def store_nodes(root):
    """Every directory named p4 whose parent is .claude: a store node, with its path from the outermost."""
    out = []
    for d, dirs, _files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git"]
        if os.path.basename(d) == "p4" and os.path.basename(os.path.dirname(d)) == ".claude":
            out.append(d)
    return sorted(out, key=len)


def size_of(path, skip=None):
    """(bytes, files) under path, .git left out, and `skip` (a nested .claude) left out."""
    b = f = 0
    for d, dirs, files in os.walk(path):
        dirs[:] = [x for x in dirs if x != ".git" and os.path.join(d, x) != skip]
        for name in files:
            try:
                b += os.path.getsize(os.path.join(d, name))
                f += 1
            except OSError:
                pass
    return b, f


def gamma(n):
    b = bin(n)[2:]
    return "0" * (len(b) - 1) + b


def ungamma(bits):
    out, i = [], 0
    while i < len(bits):
        z = 0
        while bits[i] == "0":
            z += 1
            i += 1
        out.append(int(bits[i:i + z + 1], 2))
        i += z + 1
    return out


def fib_value(depth):
    """The path digits (10)^(depth-1)1 read as Zeckendorf digits: 1 + 3 + 8 + ... = F(2*depth+1) - 1."""
    places, v = [1, 2], 0
    while len(places) < 2 * depth:
        places.append(places[-1] + places[-2])
    return sum(places[2 * k] for k in range(depth))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins", required=True, help="<p4 checkout>/lbiza/plugins")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--sessions-outer", type=int, default=1, help="sessions per level above the deepest")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    root = tempfile.mkdtemp(prefix="nested-store-")
    os.makedirs(os.path.join(root, "src"))
    with open(os.path.join(root, "src", "work.txt"), "w") as f:
        f.write("the project the outermost store belongs to\n")
    grow(args.plugins, root, args.depth, args.sessions_outer)
    report = {"root": root, "depth": args.depth, "sessions_outer": args.sessions_outer}

    # 1. the digits of every node path
    nodes = store_nodes(root)
    outer = nodes[0]
    digits = {}
    for nd in nodes:
        rel = os.path.relpath(nd, os.path.dirname(os.path.dirname(outer))).replace(os.sep, "/").split("/")[1:]
        digits[nd] = node_digits(rel)
    report["nodes"] = len(nodes)
    report["digit_strings"] = sorted(set(digits.values()), key=len)
    report["any_11"] = any("11" in s for s in digits.values())
    report["any_00"] = any("00" in s for s in digits.values())
    report["any_other_component"] = any("?" in s for s in digits.values())
    every_dir = [os.path.relpath(d, root).replace(os.sep, "/") for d, dirs, _ in os.walk(root)
                 for _ in [dirs.__setitem__(slice(None), [x for x in dirs if x != ".git"])]]
    report["p4_inside_p4"] = sum(1 for d in every_dir if "/p4/p4" in "/" + d)
    report["claude_inside_claude"] = sum(1 for d in every_dir if "/.claude/.claude" in "/" + d)
    report["values_match_F2n1_minus_1"] = all(
        int(sum(int(c) * p for c, p in zip(reversed(s), _places(len(s))))) == fib_value((len(s) + 1) // 2)
        for s in digits.values())

    # 2. a stream of node paths, one per file, by the node that owns the file, parsed with no length headers
    depth_of = {nd: (len(digits[nd]) + 1) // 2 for nd in nodes}
    stream_depths = []
    for nd in nodes:
        child = os.path.join(nd, ".claude")
        _b, files = size_of(nd, skip=child)
        stream_depths += [depth_of[nd]] * files
    tree_code = "".join(digits[nd] + "1" for nd in nodes for _ in range(size_of(nd, os.path.join(nd, ".claude"))[1]))
    values, end = read_codes(tree_code)
    inverse = {fib_value(d): d for d in range(1, args.depth + 1)}
    report["stream_items"] = len(stream_depths)
    report["tree_code_roundtrip"] = [inverse[v] for v in values] == stream_depths and end == len(tree_code)
    bare = "".join(digits[nd] for nd in nodes)                  # the digits alone, no comma
    try:
        bare_vals, bare_end = read_codes(bare)
        report["bare_digits_parse"] = f"{len(bare_vals)} codes for {len(nodes)} nodes, ended at {bare_end}/{len(bare)}"
    except ValueError as e:
        report["bare_digits_parse"] = f"refused: {e}"
    fib_depth = "".join(fib_code(d) for d in stream_depths)
    report["fib_depth_roundtrip"] = read_codes(fib_depth)[0] == stream_depths
    g = "".join(gamma(d) for d in stream_depths)
    report["gamma_roundtrip"] = ungamma(g) == stream_depths
    width = max(1, math.ceil(math.log2(args.depth + 1)))
    header = 8 + 32                                             # the width, and the item count
    n = len(stream_depths)
    path_len_field = max(1, math.ceil(math.log2(2 * args.depth)))
    report["bits_per_item"] = {
        "tree code (path digits + comma = Fibonacci code of F(2n+1)-1)": len(tree_code) / n,
        "Fibonacci code of the depth": len(fib_depth) / n,
        "Elias-gamma of the depth": len(g) / n,
        f"fixed {width}-bit depth + {header}-bit header": (width * n + header) / n,
        f"path digits + {path_len_field}-bit length field each": sum(2 * d - 1 + path_len_field for d in stream_depths) / n,
    }

    # 4. lvalue (the nested .claude/ subtree) against rvalue (the node's own content), by depth
    rows = []
    for nd in nodes:
        child = os.path.join(nd, ".claude")
        rb, rf = size_of(nd, skip=child)
        lb, lf = size_of(child) if os.path.isdir(child) else (0, 0)
        rows.append({"depth": depth_of[nd], "rvalue_bytes": rb, "rvalue_files": rf, "lvalue_bytes": lb,
                     "lvalue_files": lf, "ratio": round(lb / rb, 3) if rb else None})
    for a, b in zip(rows, rows[1:]):
        a["lvalue_is_next_l_plus_r"] = a["lvalue_bytes"] == b["lvalue_bytes"] + b["rvalue_bytes"]
        a["rvalue_over_next_lvalue"] = round(a["rvalue_bytes"] / b["lvalue_bytes"], 3) if b["lvalue_bytes"] else None
    report["by_depth"] = rows

    out = args.out or os.path.join(root, "nested-store-report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


def _places(n):
    p = [1, 2]
    while len(p) < n:
        p.append(p[-1] + p[-2])
    return p[:n]


if __name__ == "__main__":
    sys.exit(main())
