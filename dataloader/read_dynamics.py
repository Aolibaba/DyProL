from pathlib import Path

import numpy as np
import MDAnalysis as mda
from Bio.PDB import PDBParser
from MDAnalysis.analysis import align, rms
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

Amino_acid_type = [
    "ILE", "VAL", "LEU", "PHE", "CYS", "MET", "ALA", "GLY", "THR", "SER",
    "TRP", "TYR", "PRO", "HIS", "GLU", "GLN", "ASP", "ASN", "LYS", "ARG",
]
token_dict = {item: index + 1 for index, item in enumerate(Amino_acid_type)}
residue_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y'
}
SS8_ORDER = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
SS3_ORDER = ['H', 'E', 'C']
MAX_ASA = {
    'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148, 'Q': 214, 'E': 214,
    'G': 97, 'H': 216, 'I': 195, 'L': 191, 'K': 230, 'M': 203, 'F': 228,
    'P': 154, 'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165
}


def read_PDB_ensemble(pdb_id, XTC_dir):
    output_path = Path(XTC_dir)
    features = {}
    esmdiff_file = output_path / f'{pdb_id}.pdb'

    if not esmdiff_file.exists():
        print(f"Skip {pdb_id}，missing pdb file")
        return features, [], []

    u = mda.Universe(str(esmdiff_file))
    residues = u.residues.resnames
    n_frames = len(u.trajectory)
    ref = mda.Universe(str(esmdiff_file))

    rmsd_matrix = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        R = rms.RMSD(u, u, select='name CA', ref_frame=i)
        R.run()
        rmsd_matrix[i, :] = R.results.rmsd[:, 2]

    upper_tri = np.triu(rmsd_matrix)
    rmsd_matrix = upper_tri + upper_tri.T - np.diag(np.diag(upper_tri))
    np.fill_diagonal(rmsd_matrix, 0)

    ref_t = np.argmin(np.mean(rmsd_matrix, axis=-1))
    ref.trajectory[ref_t]
    align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

    atoms = u.atoms.names
    xyz_CA = np.zeros((n_frames, len(residues), 3))
    xyz_C = np.zeros((n_frames, len(residues), 3))
    xyz_N = np.zeros((n_frames, len(residues), 3))
    xyz_O = np.zeros((n_frames, len(residues), 3))

    for index, ts in enumerate(u.trajectory):
        residues_num = 0
        for num in range(len(atoms)):
            if atoms[num] == 'CA':
                xyz_CA[index][residues_num] = ts.positions[num]
            if atoms[num] == 'N':
                xyz_N[index][residues_num] = ts.positions[num]
            if atoms[num] == 'C':
                xyz_C[index][residues_num] = ts.positions[num]
            if atoms[num] == 'O':
                xyz_O[index][residues_num] = ts.positions[num]
                residues_num += 1

    features['xyz_CA'] = xyz_CA
    features['xyz_C'] = xyz_C
    features['xyz_N'] = xyz_N
    features['xyz_O'] = xyz_O
    return features, residues, rmsd_matrix


def read_xtc_ensemble(pdb_id, XTC_dir):
    output_path = Path(XTC_dir)
    features = {}
    protein_dir = output_path / f'{pdb_id}'
    xtc_file = protein_dir / 'samples.xtc'
    pdb_file = protein_dir / 'topology.pdb'

    if not xtc_file.exists() or not pdb_file.exists():
        print(f"Skip {pdb_id}，missing XTC or PDB file")
        return features, [], []

    u = mda.Universe(str(pdb_file), str(xtc_file))
    ref = mda.Universe(str(pdb_file), str(xtc_file))
    residues = u.residues.resnames
    n_frames = len(u.trajectory)

    rmsd_matrix = np.zeros((n_frames, n_frames))
    for i in range(len(u.trajectory)):
        R = rms.RMSD(u, u, select='name CA', ref_frame=i)
        R.run()
        rmsd_matrix[i, :] = R.results.rmsd[:, 2]

    upper_tri = np.triu(rmsd_matrix)
    rmsd_matrix = upper_tri + upper_tri.T - np.diag(np.diag(upper_tri))
    np.fill_diagonal(rmsd_matrix, 0)

    ref_t = np.argmin(np.mean(rmsd_matrix, axis=-1))
    ref.trajectory[ref_t]
    align.AlignTraj(u, ref, select='name CA', in_memory=True).run()

    atoms = u.atoms.names
    xyz_CA = np.zeros((len(u.trajectory), len(residues), 3))
    xyz_C = np.zeros((len(u.trajectory), len(residues), 3))
    xyz_N = np.zeros((len(u.trajectory), len(residues), 3))
    xyz_O = np.zeros((len(u.trajectory), len(residues), 3))
    xyz_sum = np.zeros((len(u.trajectory), 3))

    for index, ts in enumerate(u.trajectory):
        residues_num = 0
        for num in range(len(atoms)):
            if atoms[num] == 'CA':
                xyz_CA[index][residues_num] = ts.positions[num]
                xyz_sum[index] += ts.positions[num]
            if atoms[num] == 'N':
                xyz_N[index][residues_num] = ts.positions[num]
            if atoms[num] == 'C':
                xyz_C[index][residues_num] = ts.positions[num]
            if atoms[num] == 'O':
                xyz_O[index][residues_num] = ts.positions[num]
                residues_num += 1

    features['xyz_CA'] = xyz_CA
    features['xyz_C'] = xyz_C
    features['xyz_N'] = xyz_N
    features['xyz_O'] = xyz_O
    return features, residues, rmsd_matrix


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
    L = ca.shape[0]
    feats = []

    for gap in max_gap_list:
        if L > gap:
            feats.append(np.linalg.norm(ca[gap:] - ca[:-gap], axis=1))

    phi_sin, phi_cos = [], []
    psi_sin, psi_cos = [], []

    for i in range(1, L):
        phi = calc_dihedral(c[i - 1], n[i], ca[i], c[i])
        phi_sin.append(np.sin(phi))
        phi_cos.append(np.cos(phi))

    for i in range(L - 1):
        psi = calc_dihedral(n[i], ca[i], c[i], n[i + 1])
        psi_sin.append(np.sin(psi))
        psi_cos.append(np.cos(psi))

    feats.extend([
        np.array(phi_sin), np.array(phi_cos),
        np.array(psi_sin), np.array(psi_cos),
    ])
    return np.concatenate(feats, axis=0)


def build_feature_matrix(structure):
    xyz_CA = np.array(structure['xyz_CA'])
    xyz_N = np.array(structure['xyz_N'])
    xyz_C = np.array(structure['xyz_C'])

    all_feats = []
    for i in range(xyz_CA.shape[0]):
        all_feats.append(extract_frame_features(xyz_CA[i], xyz_N[i], xyz_C[i]))

    feature_matrix = np.stack(all_feats, axis=0)
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True) + 1e-8
    return (feature_matrix - mean) / std


def find_representative_conformations_dbscan(feature_matrix):
    pairwise = squareform(pdist(feature_matrix, metric='euclidean'))
    eps = np.percentile(pairwise[pairwise > 0], 30)
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='euclidean')
    clusters = dbscan.fit_predict(feature_matrix)

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        median_dist = np.median(pairwise[0])
        clusters = (pairwise[0] > median_dist).astype(int)
        unique_clusters = np.unique(clusters)

    representative_frames = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_center = feature_matrix[cluster_indices].mean(axis=0)
            cluster_distances = np.linalg.norm(
                feature_matrix[cluster_indices] - cluster_center, axis=1
            )
            representative_frames.append(cluster_indices[np.argmin(cluster_distances)])
    return representative_frames


def find_representative_conformations_kmeans(rmsd_matrix, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rmsd_matrix)
    representative_frames = []
    for cluster_id in range(n_cluster):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_distances = np.linalg.norm(
                rmsd_matrix[cluster_indices] - cluster_center, axis=1
            )
            representative_frames.append(cluster_indices[np.argmin(cluster_distances)])
    return representative_frames


def find_representative_conformations_hierarchical(rmsd_matrix, n_cluster):
    clustering = AgglomerativeClustering(
        n_clusters=n_cluster,
        metric='euclidean',
        linkage='average',
    )
    clusters = clustering.fit_predict(rmsd_matrix)

    representative_frames = []
    for cluster_id in range(n_cluster):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            cluster_center = rmsd_matrix[cluster_indices].mean(axis=0)
            cluster_distances = np.linalg.norm(
                rmsd_matrix[cluster_indices] - cluster_center, axis=1
            )
            representative_frames.append(cluster_indices[np.argmin(cluster_distances)])
    return representative_frames


def extract_represent(structure, represent_frames):
    represent_features = {}
    for atom_type in ['xyz_CA', 'xyz_C', 'xyz_N', 'xyz_O']:
        original_coords = structure[atom_type]
        represent_features[atom_type] = [original_coords[idx] for idx in represent_frames]
    return represent_features


def cal_nuv(represent_structure):
    xyz_CA = np.array(represent_structure['xyz_CA'])
    xyz_C = np.array(represent_structure['xyz_C'])
    xyz_N = np.array(represent_structure['xyz_N'])

    vec_C = xyz_C - xyz_CA
    vec_N = xyz_N - xyz_CA
    x_axis = vec_C / np.linalg.norm(vec_C, axis=2, keepdims=True)
    projection_Nonx = np.sum(vec_N * x_axis, axis=2, keepdims=True) * x_axis
    y_axis_ortho = vec_N - projection_Nonx
    y_axis = y_axis_ortho / np.linalg.norm(y_axis_ortho, axis=2, keepdims=True)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis, axis=2, keepdims=True)
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec

def read_dynamic(pdb_id, n_cluster, ensemble, XTC_dir, opt):
    if ensemble == 'BioEmu':
        structure, sequences, rmsd_matrix = read_xtc_ensemble(pdb_id, XTC_dir)
    else:
        structure, sequences, rmsd_matrix = read_PDB_ensemble(pdb_id, XTC_dir)

    if len(sequences) == 0:
        return np.empty((0)), np.empty((0, 3)), np.empty((0, 3, 3)), ''

    tokens = []
    fasta = ''
    for res in sequences:
        tokens.append(token_dict[res])
        fasta += residue_dict[res]

    n_frames = np.array(structure['xyz_CA']).shape[0]
    if n_frames <= n_cluster:
        represent_frames = list(range(n_frames))
        represent_structure = structure
    else:
        feature_matrix = build_feature_matrix(structure)
        represent_frames = find_representative_conformations_hierarchical(feature_matrix, n_cluster)
        represent_structure = extract_represent(structure, represent_frames)

    nuv = cal_nuv(represent_structure)
    return np.array(tokens), np.array(represent_structure['xyz_CA']), nuv, fasta
