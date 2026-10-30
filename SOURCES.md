# Grounded source inventory (Google Drive)

## Original source notebooks

- `DMD.ipynb` (83 cells): https://colab.research.google.com/drive/11NMC4X1vy_xPPFGkONXGPYOFYU1yoCBl
- `NFATC4.ipynb` (58 cells): https://colab.research.google.com/drive/1GzAWzfVj-kfvclU_4zZLETaIPfJ7arGG

Their code and markdown are copied into `reference_notebooks/` without execution outputs. The relative code and original Colab Drive paths are deliberately preserved as the research archive.

## Supplied model/graph files

- DMD trained Bayesian GAT checkpoint: https://drive.google.com/file/d/13fsQd39bl_Q3ABN3za-mVXznzJ7OPkjG/view
- DMD PPI gene edges: https://drive.google.com/file/d/1RUs42HLj7gDG6rznTJOxXjhFxhvsW_tD/view
- DMD PPI raw edges: https://drive.google.com/file/d/1-4pFrv_f7KD-eKa3IyDVLhz5NqXdScVA/view
- DMD protein-coding gene TPM expression: https://drive.google.com/file/d/1-7PstwpNomJpHicj-yUcopV3TRpcN0Al/view
- NFATC4 trained Bayesian GVP-named checkpoint: https://drive.google.com/file/d/1YolRLrag8Eyp9SjreApsbRmZcl_hYNhd/view
- NFATC4 residue graph: https://drive.google.com/file/d/11AuXATcn3kq0y9X40T3hSZLy40Jf5zWi/view
- NFATC4 AlphaFold PDB v4: https://drive.google.com/file/d/1uL7LfsLoh5kPCQSEpZ8umKOWFksmULGm/view
- NFATC4 UniProt Q14934 FASTA: https://drive.google.com/file/d/1ZYukZgUYlHxmIW2ji4ylyZ2TZfCshZDV/view
- NFATC4 ESM per-residue embeddings: https://drive.google.com/file/d/1ZeH-5o9shxtyu3nGZd9X5vn7StN_KiLe/view
- NFATC4 mutation simulation vector: https://drive.google.com/file/d/1WB5RVQ9Z_Mm8jUH9mR2w_KLLTjo7NPtR/view
- NFATC4 GO annotations: https://drive.google.com/file/d/1aQdFIO983exmBfhIgvZ_IFK0XEGJP-5X/view
- NFATC4 saliency scores: https://drive.google.com/file/d/175TJGjWiKoajQJEKRnE26gRRLX4Zmei7/view
- NFATC4 MC-dropout score standard deviation: https://drive.google.com/file/d/1Rva5RdkXx_ZmmXGL-V8sAWSoLw_mMOjB/view
- NFATC4 hidden-representation delta: https://drive.google.com/file/d/1UmvGn279gfYxngM3g321EMzfkyZRGdjX/view
- NFATC4 relative solvent accessibility: https://drive.google.com/file/d/1yl8Io79SScfSYm4ewE9ZlssAL1jRQgvK/view
- NFATC4 PyMOL highlight script: https://drive.google.com/file/d/1qnHlm4p_fa9xvRNr9nmhy3zez8vl5138/view

## Other essential Drive data NOT included (large)

The GAT training code references a large file that must be brought to the local machine separately if original-scale training is desired:

- `dmd_dmd_graph/proc/node_data/node_features.parquet` (~111 MB): https://drive.google.com/file/d/1J8tTW14Af1ubB-HKGFICywsq7XUk0WKJ/view

Further large and/or processed inputs include:

- `dmd_dmd_graph/proc/augmented_data/augmented_X_y.pt` (~5.64 GB): https://drive.google.com/file/d/1sOURAVHnxP7q4k0abKKHXdapVPc-tnXz/view
- `dmd_dmd_graph/proc/augmented_data/augmented_edge_index.pt` (~24 MB): https://drive.google.com/file/d/1NI9xZ2JAlUYfohRRs4AyGryN5nQMkz9R/view
- `dmd_dmd_graph/proc/augmented_data/augmented_node_features.parquet` (~125 MB): https://drive.google.com/file/d/1plHSeYJZqESroGaugNsDs_9pxm6AOvVl/view
- `dmd_dmd_graph/proc/esm2_150M_proj.npy` (~193 MB): https://drive.google.com/file/d/1j1i04EE2PCfpOSkZguMGLi9LX1xZOtyU/view
- `dmd_dmd_graph/proc/go_matrix.parquet` (~39 MB): https://drive.google.com/file/d/17xB1n8HNBbQxCZU3hy53mK6aDmN3Bqom/view

Source directories:

- https://drive.google.com/drive/folders/1AN-qdiEUSuxlTSEkgJ5oeuvbZkqISNlc (`dmd_dmd_graph`)
- https://drive.google.com/drive/folders/1lVxV3bzntVH0BFd_ylrYS4vsZa7omRGe (`NFATC4`)

These links are source references, not evidence the large files are downloaded to the laptop. Do not claim a successful full run without acquiring and checking the datasets.
