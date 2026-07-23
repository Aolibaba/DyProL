# DyProL

DyProL is a dynamic protein representation learning framework for residue-level protein-nucleic acid binding-site prediction.

This cleaned version keeps one supported training path only:

- dataset: `GraphBind`
- structure input: BioEmu conformational ensembles
- features: residue token, ESM/LLM embedding, MSA feature, and PSSM feature
- task: DNA-binding or RNA-binding residue prediction

![DyProL overview](fig/Fig.png)

## Dataset

Download the dataset package from Zenodo and place the resulting `Datasets/` folder at the repository root:

https://zenodo.org/records/19547616

The retained code path expects this layout:

```text
Datasets/
└── GraphBind/
    ├── DNA/
    │   ├── DNA-573_Train.txt
    │   ├── DNA-129_Test.txt
    │   ├── BioEmu/
    │   │   └── <protein_id>/
    │   │       ├── samples.xtc
    │   │       └── topology.pdb
    │   └── ESM_MSA_PSSM/
    │       ├── <protein_id>.rep_5120.npy
    │       ├── <protein_id>msa_first_row.npy
    │       ├── <protein_id>.fasta
    │       └── <protein_id>.pssm
    └── RNA/
        ├── RNA-495_Train.txt
        ├── RNA-117_Test.txt
        ├── BioEmu/
        └── ESM_MSA_PSSM/
```

Other dataset sources, static-structure mode, alternative ensemble formats, and feature-disable switches have been removed.

## Installation

Install the Python dependencies in your training environment:

```bash
pip install -r requirements.txt
```

`torch-scatter` wheels are tied to the installed PyTorch and CUDA versions. If the generic installation fails, install the matching wheel from the official PyTorch Geometric wheel index for your environment.

This project expects the NumPy 1.x ABI (`numpy<2`) because PyTorch, MDAnalysis, and other compiled scientific packages must be built against a compatible NumPy version.

## Training

Train on GraphBind DNA:

```bash
python train.py --ligand DNA
```

Train on GraphBind RNA:

```bash
python train.py --ligand RNA
```

For a quick data-loading check:

```bash
python train.py --ligand DNA --max_samples 2
```

## Options

The remaining runtime options are intentionally narrow:

- `--ligand`: `DNA` or `RNA`
- `--n_cluster`: representative BioEmu conformations selected per protein
- `--device`: training device, for example `cpu` or `cuda:0`
- `--batch_size`: proteins per batch
- `--emb_dims`: hidden feature dimension
- `--n_layers_structure`: number of geometric attention layers
- `--lr`: learning rate
- `--seed`: random seed
- `--checkpoints_dir`: optional output directory
- `--max_samples`: optional debug sample limit

The model always loads token, ESM/LLM, MSA, and PSSM features and fuses them by concatenation.
