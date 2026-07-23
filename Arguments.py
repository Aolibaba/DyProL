import argparse

parser = argparse.ArgumentParser(
    description="Train DyProL on the GraphBind dynamic benchmark with token, LLM, MSA, and PSSM features."
)

parser.add_argument(
    "--ligand",
    type=str,
    default="DNA",
    choices=["DNA", "RNA"],
    help="GraphBind ligand type.",
)

parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default=None,
    help="Where to save logs and checkpoints. Defaults to Checkpoints/GraphBind/<ligand>.",
)

parser.add_argument(
    "--n_cluster",
    type=int,
    default=12,
    help="Number of representative conformations selected from the BioEmu ensemble.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to train on, for example cpu or cuda:0.",
)

parser.add_argument(
    "--n_layers_structure",
    type=int,
    default=4,
    help="Number of geometric attention layers.",
)

parser.add_argument(
    "--emb_dims",
    type=int,
    default=64,
    help="Hidden feature dimension.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Number of proteins in a batch.",
)

parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional debug limit for the number of samples loaded per split.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed.",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    help="Learning rate.",
)

parser.add_argument(
    "--msa_dim",
    type=int,
    default=256,
    help="MSA feature dimension per residue.",
)

parser.add_argument(
    "--llm_dim",
    type=int,
    default=5120,
    help="ESM2 residue embedding dimension.",
)

parser.add_argument(
    "--pssm_dim",
    type=int,
    default=20,
    help="PSSM feature dimension per residue.",
)
