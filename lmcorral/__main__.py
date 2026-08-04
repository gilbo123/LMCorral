"""Enables `python -m lmcorral`, which works from a source checkout with no install."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
