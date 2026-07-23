# Datasets

This directory is intentionally left empty in the source tree.

Download the data package from Zenodo:

https://zenodo.org/records/19547616

This cleaned codebase only uses the GraphBind BioEmu path:

```text
Datasets/
└── GraphBind/
    ├── DNA/
    │   ├── DNA-573_Train.txt
    │   ├── DNA-129_Test.txt
    │   ├── BioEmu/
    │   └── ESM_MSA_PSSM/
    └── RNA/
        ├── RNA-495_Train.txt
        ├── RNA-117_Test.txt
        ├── BioEmu/
        └── ESM_MSA_PSSM/
```

The retained training pipeline requires BioEmu conformational ensembles plus ESM/LLM, MSA, and PSSM residue features.
