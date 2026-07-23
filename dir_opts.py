from pathlib import Path


def dir_opts(args, project_root):
    graphbind_dir = Path(project_root) / "Datasets" / "GraphBind" / args.ligand
    feature_dir = graphbind_dir / "ESM_MSA_PSSM"

    return {
        "ensemble_dir": str(graphbind_dir / "BioEmu"),
        "ESM_AF_dir": str(feature_dir),
        "pssm_dir": str(feature_dir),
    }
