"""The p4 store's layout: each plugin writes in a directory of its own under `.claude/p4/`.

p4-recall keeps `recall/store/` and `recall/models/`, p4-temportal keeps `temportal/memory/` and `temportal/src/`.
A store an older plugin wrote flat has them at the top, until the next session start moves it. `sub()` reads the
layout once per store: flat when the store holds a flat directory and no plugin directory, else the new one. This
mirrors lbiza/plugins/*/p4conf.py LAYOUT, which the plugins use.
"""

import os
from pathlib import Path

LAYOUT = {
    "memory": "temportal/memory",
    "scratchpad": "temportal/scratchpad",
    "tmp": "temportal/tmp",
    "src": "temportal/src",
    "store": "recall/store",
    "models": "recall/models",
}


def is_flat(base):
    """True for a store written by the older plugins: a flat directory at its top, and no plugin directory."""
    if any(os.path.isdir(os.path.join(base, d)) for d in {v.split("/")[0] for v in LAYOUT.values()}):
        return False
    return any(os.path.isdir(os.path.join(base, k)) for k in LAYOUT)


def sub(p4, name, *rest):
    """Where `name` lives under the store `p4`, in either layout. Returns a Path when given one, else a str."""
    base = os.fspath(p4)
    top = os.path.join(base, name) if is_flat(base) else os.path.join(base, *LAYOUT.get(name, name).split("/"))
    out = os.path.join(top, *rest)
    return Path(out) if isinstance(p4, Path) else out
