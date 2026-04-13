from pathlib import Path
def dir_opts(args, PROJECT_ROOT):
    dir_opts = {}

    datasets_dir = Path('./Datasets')
    if args.data_source == 'dyprol':
        dir_opts['PDB_dir'] = str(datasets_dir / 'DyProL' /args.ligand / 'PDB')
        dir_opts['ensemble_dir'] = str(datasets_dir / 'DyProL' / args.ligand / args.ensemble)

    elif args.data_source == 'graphbind':
        dir_opts['PDB_dir'] = str(datasets_dir / 'GraphBind' / args.ligand / 'PDB')
        dir_opts['pssm_dir'] = str(datasets_dir / 'GraphBind' / args.ligand / 'ESM_MSA_PSSM')
        dir_opts['hhm_dir'] = str(datasets_dir / 'GraphBind' / args.ligand / 'ESM_MSA_PSSM')
        dir_opts['XTC_dir'] = str(datasets_dir / 'GraphBind' / args.ligand / 'BioEmu')
        dir_opts['ESM_AF_dir'] = str(datasets_dir / 'GraphBind' / args.ligand / 'ESM_MSA_PSSM')
    return dir_opts