import argparse

parser = argparse.ArgumentParser(description="Network parameters")

parser.add_argument(
    "--ligand",
    type=str,
    default='DNA',
    choices=['DNA', 'RNA'],
    help="Ligand type.",
)

parser.add_argument(
    '--data_source',
    type=str,
    default='dyprol',
    choices=['graphbind', 'dyprol'],
    help='Which dataset to use.',
)

parser.add_argument(
    '--dynamic_mode',
    type=str,
    default='dynamic',
    choices=['dynamic', 'static'],
    help='Whether to use dynamic pdbs.',
)

parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default="Checkpoints/RNA",
    help="Where the log and model save",
)

parser.add_argument(
    '--ensemble',
    type=str,
    default='ESMFlow',
    choices=['BioEmu', 'ESMFlow', 'ALphaFlow', 'ESMDiff'],
    help='Which ensemble method to use.',
)

parser.add_argument(
    '--use_llm',
    type=bool,
    default=False,
    help='Whether to use ESM for llm or not',
)

parser.add_argument(
    '--use_pssm',
    type=bool,
    default=False,
    help='Whether to use PSSM for llm or not',
)

parser.add_argument(
    '--use_msa',
    type=bool,
    default=False,
    help='Whether to use msa features'
)

parser.add_argument(
    '--use_token',
    type=bool,
    default=True,
    help='Whether to use msa features'
)

parser.add_argument(
    '--msa_dim',
    type=int,
    default=256,
    help='msa feature dimension per residue'
)

parser.add_argument(
    '--llm_dim',
    type=int,
    default=5120,
    help='ESM2 residue embedding dimension'
)

parser.add_argument(
    '--pssm_dim',
    type=int,
    default=20,
    help='PSSM residue embedding dimension'
)

parser.add_argument(
    '--fusion_type',
    type=str,
    default='concat',
    help='Feature fusion type: concat or gate'
)

parser.add_argument(
    "--n_cluster",
    type=int,
    default=12,
    help="Number of clusters for dynamic pdbs for the method of KMeans or hierarchical clustering",
)

parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Which gpu/cpu to train on"
)

parser.add_argument(
    "--n_layers_structure",
    type=int,
    default=4,
    help="Number of convolutional layers"
)

parser.add_argument(
    "--emb_dims",
    type=int,
    default=64,
    help="Number features of hidden layers ",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Number of proteins in a batch"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42, help="Random seed")

parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    help="Learning rate",
)