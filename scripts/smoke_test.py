"""CPU smoke test: synthetic GAT demo + NFATC4 saved-array analysis (if assets exist)."""

from dmd_nfatc4.cli import main
from dmd_nfatc4.config import DEFAULT_ASSETS

if __name__ == "__main__":
    rc = main(["demo", "--epochs", "20"])
    if (DEFAULT_ASSETS / "nfatc4" / "saliency_vector.npy").exists():
        rc |= main(["analyze-nfatc4", "--from-saved-arrays"])
        rc |= main(["inspect-checkpoint"])
    else:
        print("assets/ not found: skipped NFATC4 analysis and checkpoint checks")
    raise SystemExit(rc)
