import os
import random
from pathlib import Path

import numpy as np
import torch
import tqdm

from .read_dynamics import read_dynamic

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "Datasets"

tensor = torch.FloatTensor
inttensor = torch.LongTensor


def extract_topology(x1, x2=None, num_nn=64, cutoff=20.0):
    if x2 is None:
        x2 = x1

    distances = np.linalg.norm(x1[None, :, :] - x2[:, None, :], axis=2)

    knn = min(num_nn, distances.shape[1])
    ids_topk = np.argpartition(distances, knn - 1, axis=1)[:, :knn]
    distances_topk = np.take_along_axis(distances, ids_topk, axis=1)

    ids_topk = ids_topk + 1
    ids_topk[distances_topk > cutoff] = 0

    padding_num = max(0, num_nn - distances.shape[1])
    return np.pad(ids_topk, ((0, 0), (0, padding_num)), constant_values=0)


def pack(token=None, xyz=None, nuv=None, llm=None, topk=None, y=None, msa=None, pssm=None, pdb_id=None):
    return {
        "token": token,
        "xyz": xyz,
        "nuv": nuv,
        "y": y,
        "llm": llm,
        "topk": topk,
        "msa": msa,
        "pssm": pssm,
        "pdb_id": pdb_id,
    }


def empty_pack():
    return pack(
        token=inttensor([]),
        xyz=tensor([]),
        nuv=tensor([]),
        llm=tensor([]),
        topk=inttensor([]),
        y=inttensor([]),
        msa=tensor([]),
        pssm=tensor([]),
        pdb_id="",
    )


def feature_id_for(name):
    return name[:-1] if len(name) > 6 else name


def validate_feature(feature, name, feature_name, expected_len, expected_dim):
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(f"{feature_name} for {name} must be 2D, got shape {feature.shape}.")
    if feature.shape[0] != expected_len:
        raise ValueError(
            f"{feature_name} length mismatch for {name}: seq={expected_len}, feature={feature.shape[0]}."
        )
    if feature.shape[1] != expected_dim:
        raise ValueError(
            f"{feature_name} dim mismatch for {name}: expected={expected_dim}, feature={feature.shape[1]}."
        )
    return feature


def load_esm_feature(name, sequence, opt):
    llm_path = os.path.join(opt.dir_opts["ESM_AF_dir"], name + ".rep_5120.npy")
    feature = np.load(llm_path)[1:-1]
    return validate_feature(feature, name, "ESM", len(sequence), opt.llm_dim)


def load_msa_af_feature(name, sequence, opt):
    msa_af_path = os.path.join(opt.dir_opts["ESM_AF_dir"], name + "msa_first_row.npy")
    feature = np.load(msa_af_path)
    return validate_feature(feature, name, "MSA", len(sequence), opt.msa_dim)


def load_esm_seq(name, opt):
    fasta_path = os.path.join(opt.dir_opts["ESM_AF_dir"], name + ".fasta")
    with open(fasta_path, "r") as fasta_file:
        lines = fasta_file.readlines()
    return lines[1].strip()


def load_pssm_feature(name, sequence, opt):
    pssm_path = os.path.join(opt.dir_opts["pssm_dir"], f"{name}.pssm")
    if not os.path.exists(pssm_path):
        raise FileNotFoundError(f"PSSM file missing for {name}: {pssm_path}")

    pssm_rows = []
    pssm_seq = []
    with open(pssm_path, "r") as pssm_file:
        for line in pssm_file:
            parts = line.strip().split()
            if len(parts) < 22 or not parts[0].isdigit() or len(parts[1]) != 1:
                continue
            scores = [float(x) for x in parts[2:22]]
            pssm_seq.append(parts[1])
            pssm_rows.append(scores)

    if not pssm_rows:
        raise ValueError(f"No valid PSSM rows found for {name}.")

    if "".join(pssm_seq) != sequence:
        raise ValueError(f"PSSM sequence mismatch for {name}.")

    return validate_feature(pssm_rows, name, "PSSM", len(sequence), opt.pssm_dim)


def compute_pdb(protein, opt):
    name, sequence, label = protein[0], protein[1], protein[2]
    label = [int(item) for item in label]
    feature_id = feature_id_for(name)

    token_, xyz_, nuv_, fasta_ = read_dynamic(name, opt.n_cluster, opt.dir_opts["ensemble_dir"])
    if fasta_ == "":
        print(f"Dynamic pdb reading failed for {name}.")
        return empty_pack()
    if sequence != fasta_:
        print(f"Sequence string mismatch for dynamic structure of ID {name}.")
        return empty_pack()
    if len(label) != len(fasta_):
        print(f"Label length mismatch for {name}: label={len(label)}, structure={len(fasta_)}.")
        return empty_pack()

    try:
        llm_ = load_esm_feature(feature_id, sequence, opt)
        fasta_llm = load_esm_seq(feature_id, opt)
        if fasta_ != fasta_llm:
            raise ValueError(f"ESM sequence mismatch for {name}.")
        msa_ = load_msa_af_feature(feature_id, sequence, opt)
        pssm_ = load_pssm_feature(feature_id, sequence, opt)
    except Exception as exc:
        print(f"Feature loading failed for {name}: {exc}")
        return empty_pack()

    topk = np.array([extract_topology(xyz_[k]) for k in range(xyz_.shape[0])])

    return pack(
        token=inttensor(token_),
        xyz=tensor(xyz_),
        nuv=tensor(nuv_),
        llm=tensor(llm_),
        topk=inttensor(topk),
        y=inttensor(label),
        msa=tensor(msa_),
        pssm=tensor(pssm_),
        pdb_id=name,
    )


def load_pdb(proteins, opt):
    print("Loading GraphBind BioEmu ensembles with ESM, MSA, and PSSM features.")
    pdbs = []
    for protein in tqdm.tqdm(proteins):
        item = compute_pdb(protein, opt)
        if len(item["token"]) >= 1:
            pdbs.append(item)
    return pdbs


def collate_multiconf(batch):
    meta = {}
    keys = batch[0].keys()
    sample_infos = []
    total_residues_all_confs = 0

    for sample in batch:
        k_current = sample["xyz"].shape[0]
        n_residues = sample["xyz"].shape[1]
        sample_infos.append({
            "k": k_current,
            "n": n_residues,
            "total_conformation_residues": k_current * n_residues,
        })
        total_residues_all_confs += k_current * n_residues

    adjustment_map = build_adjustment_map(sample_infos, total_residues_all_confs)

    for key in keys:
        if key in ["llm", "msa", "pssm"]:
            feat_tensors = []
            for sample, info in zip(batch, sample_infos):
                feat_tensors.append(sample[key].repeat(info["k"], 1))
            meta[key] = torch.cat(feat_tensors, dim=0)
        elif key == "y":
            meta[key] = torch.cat([sample[key] for sample in batch], dim=0)
        elif key == "token":
            token_tensors = []
            for sample, info in zip(batch, sample_infos):
                token_tensors.append(sample[key].repeat(info["k"]))
            meta[key] = torch.cat(token_tensors, dim=0)
        elif key in ["xyz", "nuv", "topk"]:
            concatenated_tensors = []
            for sample in batch:
                k, n = sample[key].shape[0], sample[key].shape[1]
                concatenated_tensors.append(sample[key].reshape(k * n, *sample[key].shape[2:]))
            meta[key] = torch.cat(concatenated_tensors, dim=0)
        elif key == "pdb_id":
            meta[key] = [sample[key] for sample in batch]

    meta["xyz_batch"] = torch.zeros(total_residues_all_confs, dtype=torch.int64)
    meta["amino_acid_batch"] = torch.zeros(total_residues_all_confs, dtype=torch.int64)

    current_idx = 0
    amino_acid_counter = 0
    for i, info in enumerate(sample_infos):
        k, n = info["k"], info["n"]
        meta["xyz_batch"][current_idx:current_idx + info["total_conformation_residues"]] = i
        for amino_idx in range(n):
            start_idx = current_idx + amino_idx
            for conf_idx in range(k):
                global_idx = start_idx + conf_idx * n
                meta["amino_acid_batch"][global_idx] = amino_acid_counter
            amino_acid_counter += 1
        current_idx += info["total_conformation_residues"]

    meta["cross_topk"] = build_cross_topk(sample_infos)
    meta["adjusted_topk"] = adjust_topk_fast(meta["topk"], adjustment_map)
    return meta


def build_cross_topk(sample_infos):
    total_residues_all_confs = sum(info["total_conformation_residues"] for info in sample_infos)
    max_k = max(info["k"] for info in sample_infos)
    cross_topk = torch.zeros(total_residues_all_confs, max_k, dtype=torch.long)
    current_idx = 0

    for info in sample_infos:
        k, n = info["k"], info["n"]
        indices = torch.arange(current_idx, current_idx + k * n).reshape(k, n)
        for residue_pos in range(n):
            residue_conf_indices = indices[:, residue_pos]
            neighbor_indices_1based = residue_conf_indices + 1
            for global_idx in residue_conf_indices:
                cross_topk[global_idx, :k] = neighbor_indices_1based
        current_idx += k * n
    return cross_topk


def build_adjustment_map(sample_infos, total_residues):
    adjustment_map = torch.zeros(total_residues, dtype=torch.long)
    current_idx = 0
    for info in sample_infos:
        k, n = info["k"], info["n"]
        for conf_idx in range(k):
            conf_start = current_idx + conf_idx * n
            for local_pos in range(n):
                adjustment_map[conf_start + local_pos] = conf_start
        current_idx += info["total_conformation_residues"]
    return adjustment_map


def adjust_topk_fast(topk, adjustment_map):
    total_residues, num_nn = topk.shape
    adjusted_topk = torch.zeros((total_residues + 1, num_nn), dtype=topk.dtype)
    adjusted_topk[0] = 1
    valid_mask = topk > 0
    adjustment_expanded = adjustment_map.unsqueeze(1).expand(-1, num_nn)
    adjusted_topk[1:] = torch.where(valid_mask, topk + adjustment_expanded, topk)
    return adjusted_topk


def read_graphbind_split(path, step):
    with open(path, "r") as split_file:
        lines = split_file.readlines()

    samples = []
    for i in range(0, len(lines), step):
        query_id = lines[i].strip()[1:]
        query_seq = lines[i + 1].strip()
        query_anno = lines[i + 2].strip()
        samples.append([query_id, query_seq, query_anno])
    return samples


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = self.create_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.subset == "train",
            collate_fn=collate_multiconf,
        )

    def create_dataset(self):
        base_dir = DATASETS_DIR / "GraphBind" / self.opt.ligand
        if self.opt.ligand == "RNA":
            train_file = base_dir / "RNA-495_Train.txt"
            test_file = base_dir / "RNA-117_Test.txt"
        else:
            train_file = base_dir / "DNA-573_Train.txt"
            test_file = base_dir / "DNA-129_Test.txt"

        if self.opt.subset in ["train", "val"]:
            samples = read_graphbind_split(train_file, step=4)
            random.seed(self.opt.seed)
            random.shuffle(samples)
            val_size = int(len(samples) * 0.1)
            if self.opt.subset == "train":
                samples = samples[val_size:]
            else:
                samples = samples[:val_size]
        else:
            samples = read_graphbind_split(test_file, step=3)

        if self.opt.max_samples is not None:
            samples = samples[:self.opt.max_samples]

        loaded_pdbs = load_pdb(samples, self.opt)
        print(self.opt.subset, "Total: {}".format(len(samples)), "Loaded: {}".format(len(loaded_pdbs)))
        return loaded_pdbs

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data
