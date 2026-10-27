"""Moved to phi_rank; this name stays so `python -m phi_prune.p4rank` and older vlds plugins keep working."""

import sys

from phi_rank import *  # noqa: F401,F403
from phi_rank import main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
