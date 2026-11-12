"""The p4 store's layout: each plugin writes in a directory of its own under `.claude/p4/`.

p4-phi keeps `phi/store/` and `phi/models/`, recall being part of it since p4-phi 0.7.0, and p4-temportal keeps
`temportal/memory/` and `temportal/src/`. Older stores have store/ and models/ under `recall/`, written while p4-recall
was a plugin of its own, or flat at the top, until the next session start moves them. `sub()` reads the layout once
per store: the newest one the store holds, else the newest of all. This mirrors lbiza/plugins/*/p4conf.py LAYOUT
and OLDER, which the plugins use.
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
