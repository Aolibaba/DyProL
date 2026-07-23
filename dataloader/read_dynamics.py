from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align, rms
from sklearn.cluster import AgglomerativeClustering

Amino_acid_type = [
    "ILE", "VAL", "LEU", "PHE", "CYS", "MET", "ALA", "GLY", "THR", "SER",
    "TRP", "TYR", "PRO", "HIS", "GLU", "GLN", "ASP", "ASN", "LYS", "ARG",
]
token_dict = {item: index + 1 for index, item in enumerate(Amino_acid_type)}
residue_dict = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def read_xtc_ensemble(pdb_id, ensemble_dir):
    protein_dir = Path(ensemble_dir) / pdb_id
    xtc_file = protein_dir / "samples.xtc"
    pdb_file = protein_dir / "topology.pdb"

    if not xtc_file.exists() or not pdb_file.exists():
        print(f"Skip {pdb_id}, missing BioEmu samples.xtc or topology.pdb.")
        return {}, [], None

    universe = mda.Universe(str(pdb_file), str(xtc_file))
    reference = mda.Universe(str(pdb_file), str(xtc_file))
    residues = universe.residues.resnames
    n_frames = len(universe.trajectory)

    rmsd_matrix = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        rmsd_runner = rms.RMSD(universe, universe, select="name CA", ref_frame=i)
        rmsd_runner.run()
        rmsd_matrix[i, :] = rmsd_runner.results.rmsd[:, 2]

    upper_tri = np.triu(rmsd_matrix)
    rmsd_matrix = upper_tri + upper_tri.T - np.diag(np.diag(upper_tri))
    np.fill_diagonal(rmsd_matrix, 0)

    reference_frame = np.argmin(np.mean(rmsd_matrix, axis=-1))
    reference.trajectory[reference_frame]
    align.AlignTraj(universe, reference, select="name CA", in_memory=True).run()

    atoms = universe.atoms.names
    xyz_ca = np.zeros((n_frames, len(residues), 3))
    xyz_c = np.zeros((n_frames, len(residues), 3))
    xyz_n = np.zeros((n_frames, len(residues), 3))

    for frame_index, ts in enumerate(universe.trajectory):
        residue_index = 0
        for atom_index, atom_name in enumerate(atoms):
            if atom_name == "CA":
                xyz_ca[frame_index][residue_index] = ts.positions[atom_index]
            elif atom_name == "N":
                xyz_n[frame_index][residue_index] = ts.positions[atom_index]
            elif atom_name == "C":
                xyz_c[frame_index][residue_index] = ts.positions[atom_index]
            elif atom_name == "O":
                residue_index += 1

    return {
        "xyz_CA": xyz_ca,
        "xyz_C": xyz_c,
        "xyz_N": xyz_n,
    }, residues, rmsd_matrix


def calc_dihedral(p0, p1, p2, p3):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def extract_frame_features(ca, n, c, max_gap_list=(1, 2, 4, 8)):
    length = ca.shape[0]
    feats = []

    for gap in max_gap_list:
        if length > gap:
            feats.append(np.linalg.norm(ca[gap:] - ca[:-gap], axis=1))

    phi_sin, phi_cos = [], []
    psi_sin, psi_cos = [], []

    for i in range(1, length):
        phi = calc_dihedral(c[i - 1], n[i], ca[i], c[i])
        phi_sin.append(np.sin(phi))
        phi_cos.append(np.cos(phi))

    for i in range(length - 1):
        psi = calc_dihedral(n[i], ca[i], c[i], n[i + 1])
        psi_sin.append(np.sin(psi))
        psi_cos.append(np.cos(psi))

    feats.extend([
        np.array(phi_sin),
        np.array(phi_cos),
        np.array(psi_sin),
        np.array(psi_cos),
    ])
    return np.concatenate(feats, axis=0)


def build_feature_matrix(structure):
    xyz_ca = np.array(structure["xyz_CA"])
    xyz_n = np.array(structure["xyz_N"])
    xyz_c = np.array(structure["xyz_C"])

    feature_matrix = np.stack([
        extract_frame_features(xyz_ca[i], xyz_n[i], xyz_c[i])
        for i in range(xyz_ca.shape[0])
    ], axis=0)
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True) + 1e-8
    return (feature_matrix - mean) / std


def find_representative_conformations(feature_matrix, n_cluster):
    clustering = AgglomerativeClustering(
        n_clusters=n_cluster,
        metric="euclidean",
        linkage="average",
    )
    clusters = clustering.fit_predict(feature_matrix)

    representative_frames = []
    for cluster_id in range(n_cluster):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_center = feature_matrix[cluster_indices].mean(axis=0)
            cluster_distances = np.linalg.norm(
                feature_matrix[cluster_indices] - cluster_center,
                axis=1,
            )
            representative_frames.append(cluster_indices[np.argmin(cluster_distances)])
    return representative_frames


def extract_representative_structure(structure, representative_frames):
    represent_features = {}
    for atom_type in ["xyz_CA", "xyz_C", "xyz_N"]:
        original_coords = structure[atom_type]
        represent_features[atom_type] = [original_coords[idx] for idx in representative_frames]
    return represent_features


def cal_nuv(represent_structure):
    xyz_ca = np.array(represent_structure["xyz_CA"])
    xyz_c = np.array(represent_structure["xyz_C"])
    xyz_n = np.array(represent_structure["xyz_N"])

    vec_c = xyz_c - xyz_ca
    vec_n = xyz_n - xyz_ca
    x_axis = vec_c / (np.linalg.norm(vec_c, axis=2, keepdims=True) + 1e-8)
    projection_nonx = np.sum(vec_n * x_axis, axis=2, keepdims=True) * x_axis
    y_axis_ortho = vec_n - projection_nonx
    y_axis = y_axis_ortho / (np.linalg.norm(y_axis_ortho, axis=2, keepdims=True) + 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis, axis=2, keepdims=True) + 1e-8)
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def read_dynamic(pdb_id, n_cluster, ensemble_dir):
    structure, residues, _ = read_xtc_ensemble(pdb_id, ensemble_dir)

    if len(residues) == 0:
        return np.empty((0)), np.empty((0, 3)), np.empty((0, 3, 3)), ""

    tokens = []
    fasta = ""
    for residue in residues:
        tokens.append(token_dict[residue])
        fasta += residue_dict[residue]

    n_frames = np.array(structure["xyz_CA"]).shape[0]
    if n_frames <= n_cluster:
        representative_structure = structure
    else:
        feature_matrix = build_feature_matrix(structure)
        representative_frames = find_representative_conformations(feature_matrix, n_cluster)
        representative_structure = extract_representative_structure(structure, representative_frames)

    nuv = cal_nuv(representative_structure)
    return np.array(tokens), np.array(representative_structure["xyz_CA"]), nuv, fasta
