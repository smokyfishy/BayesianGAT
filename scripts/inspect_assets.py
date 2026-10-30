"""Inspect local assets and strictly load both original checkpoints."""

import sys

from dmd_nfatc4.cli import main

if __name__ == "__main__":
    rc = main(["inspect-assets", *sys.argv[1:]])
    rc |= main(["inspect-checkpoint", *sys.argv[1:]])
    raise SystemExit(rc)
