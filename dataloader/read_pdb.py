import os
from numpy import linalg as LA

import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

Amino_acid_type = [
    "ILE", "VAL", "LEU", "PHE", "CYS", "MET", "ALA", "GLY", "THR", "SER",
    "TRP", "TYR", "PRO", "HIS", "GLU", "GLN", "ASP", "ASN", "LYS", "ARG",
]
token_dict = {item: index + 1 for index, item in enumerate(Amino_acid_type)}


def calc_local_frame(N_loc, CA_loc, C_loc):
    u_i = (N_loc - CA_loc) / (LA.norm(N_loc - CA_loc) + 1e-8)
    t_i = (C_loc - CA_loc) / (LA.norm(C_loc - CA_loc) + 1e-8)
    n_i = np.cross(u_i, t_i)
    n_i /= LA.norm(n_i)
    v_i = np.cross(n_i, u_i)
    v_i /= LA.norm(v_i)
    return np.row_stack([u_i, n_i, v_i])


def read_PDB_compatible(in_pdb_dir, pdb_name, label):
    pdb_file = os.path.join(in_pdb_dir, pdb_name)
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_name, pdb_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse {pdb_file}: {e}")
        return np.empty((0)), np.empty((0, 3)), np.empty((0, 3, 3)), '', 0, ''

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield, _, _ = residue.id
                if hetfield.strip() != '':
                    continue
                if not is_aa(residue, standard=True):
                    continue
                if residue.get_resname() not in Amino_acid_type:
                    continue
                residues.append(residue)

    seq = ''
    xyz = []
    nuv = []
    tokens = []
    sequence_pdb = ''
    label_pdb = []

    for index, residue in enumerate(residues):
        atoms = residue.child_dict
        if 'CA' not in atoms or 'C' not in atoms or 'N' not in atoms:
            continue

        CA_loc = atoms['CA'].coord
        N_loc = atoms['N'].coord
        C_loc = atoms['C'].coord

        sequence_pdb += seq1(residue.get_resname())
        if index > len(label) - 1:
            print(f"[ERROR] Label length mismatch for {pdb_name}.")
            return np.empty((0)), np.empty((0, 3)), np.empty((0, 3, 3)), '', 0, ''
        label_pdb.append(label[index])

        frame = calc_local_frame(N_loc, CA_loc, C_loc)
        resname = residue.get_resname()
        seq += seq1(resname)
        xyz.append(CA_loc)
        nuv.append(frame)
        tokens.append(token_dict[residue.get_resname()])

    if len(xyz) == 0:
        return np.empty((0)), np.empty((0, 3)), np.empty((0, 3, 3)), '', 0, ''

    return np.array(tokens), np.stack(xyz), np.stack(nuv), seq, label_pdb, sequence_pdb
