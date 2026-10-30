"""Verify local files against FILE_SHA256SUMS.txt (thin wrapper around the CLI)."""

from dmd_nfatc4.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["verify-hashes"]))
