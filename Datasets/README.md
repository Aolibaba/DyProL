# Datasets

This directory is intentionally left empty in the GitHub repository.

The complete benchmark data package for DyProL is distributed separately through Zenodo and is not tracked in version control because of its size.
To run the code, download the dataset archive from Zenodo https://zenodo.org/records/19547616, extract it, and place the resulting `Datasets/` folder at the repository root so that the expected file paths resolve correctly.

## What is included in the Zenodo release

The Zenodo archive contains the full set of data resources required for DyProL, including:

- `DyProL/`: the main dynamic-ensemble benchmark used in this work.
- `GraphBind/`: the public benchmark used for cross-dataset evaluation and comparison with prior methods.
- `DNA/` and `RNA/` splits for both benchmarks.
- Structural inputs, conformational ensembles, and sequence-derived feature files required by the code.