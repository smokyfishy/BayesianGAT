"""DMD gene-graph Bayesian GAT and NFATC4 residue-graph research package.

Two clearly separated tracks live in this package:

* ``legacy``    -- faithful re-implementations of the original 2025 Colab code
                   (``reference_notebooks/``), kept so the saved checkpoints can be
                   reloaded and the original analysis re-run. Known methodological
                   problems (label leakage, augment-before-split, simulated labels)
                   are preserved *on purpose* and flagged at runtime.
* ``corrected`` -- leakage-free evaluation protocols built on the same model
                   families. See ``docs/LEGACY_VS_CORRECTED.md``.

Nothing in this package is a clinical tool.
"""

__version__ = "0.1.0"
