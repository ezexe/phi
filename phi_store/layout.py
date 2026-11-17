"""The store's layout: each plugin writes in a folder of its own under `.claude/p4/`.

p4-phi keeps `phi/store/` and `phi/models/`, and p4-temportal keeps `temportal/memory/`, `temportal/scratchpad/`,
`temportal/tmp/` and `temportal/src/`. A store an older plugin wrote holds store/ and models/ under `recall/`, or
flat at its top, until the plugins' next session start moves them. `sub()` and `rel()` find the layout the store
holds: the newest one it holds, else the newest of all. LAYOUT holds the places of
lbiza/plugins/p4-phi/scripts/p4conf.py's LAYOUT that phi reads, and OLDER the older places of store/ and models/,
newest first.
"""

import os
from pathlib import Path

LAYOUT = {
    "memory": "temportal/memory",
    "scratchpad": "temportal/scratchpad",
    "tmp": "temportal/tmp",
    "src": "temportal/src",
    "store": "phi/store",
    "models": "phi/models",
}
# store/ and models/ in each older layout, newest first
OLDER = ({"store": "recall/store", "models": "recall/models"}, {"store": "store", "models": "models"})


def _places(base):
    """{name: path relative to the store} in the layout the store at `base` holds."""
    if is_flat(base):
        return {k: k for k in LAYOUT}
    for kinds in ({"store": LAYOUT["store"], "models": LAYOUT["models"]}, *OLDER):
        if any(os.path.isdir(os.path.join(base, *v.split("/"))) for v in kinds.values()):
            return {**LAYOUT, **kinds}
    return dict(LAYOUT)


def is_flat(base):
    """True for a store written by the oldest plugins: a flat directory at its top, and no plugin directory."""
    if any(os.path.isdir(os.path.join(base, d)) for d in ("phi", "recall", "temportal")):
        return False
    return any(os.path.isdir(os.path.join(base, k)) for k in LAYOUT)


def rel(p4, name):
    """Where `name` lives in the store `p4`, relative to it with forward slashes: the path .gitignore names."""
    return _places(os.fspath(p4)).get(name, name)


def sub(p4, name, *rest):
    """Where `name` lives under the store `p4`, in the layout it holds. Returns a Path when given one, else a str."""
    base = os.fspath(p4)
    out = os.path.join(base, *rel(base, name).split("/"), *rest)
    return Path(out) if isinstance(p4, Path) else out
