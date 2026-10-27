"""Moved to phi_serve; this name stays so `python -m phi_prune.p4serve` and older vlds plugins keep working."""

import sys

from phi_serve import *  # noqa: F401,F403
from phi_serve import main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
