"""Moved to phi_store; this name stays so `python -m phi_prune.p4` and older vlds plugins keep working."""

import sys

from phi_store import *  # noqa: F401,F403
from phi_store import main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
