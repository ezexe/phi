"""Moved to phi_model; this name stays so `python -m phi_prune.p4model` and older vlds plugins keep working."""

import sys

from phi_model import *  # noqa: F401,F403
from phi_model import main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
