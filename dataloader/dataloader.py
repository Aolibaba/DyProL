import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import tqdm
from joblib import Parallel, delayed, cpu_count

from .read_dynamics import read_dynamic
from .read_pdb import read_PDB_compatible

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / 'Datasets'

tensor = torch.FloatTensor
inttensor = torch.LongTensor


def extract_topology(X1, X2=None, num_nn=64, cutoff=20.0):
    if X2 is None:
        X2 = X1

    R = X1[None, :, :] - X2[:, None, :]
    D = np.linalg.norm(R, axis=2)

    knn = min(num_nn, D.shape[1])
    ids_topk = np.argpartition(D, knn - 1, axis=1)[:, :knn]
    distances_topk = np.take_along_axis(D, ids_topk, axis=1)

    ids_topk = ids_topk + 1
    ids_topk[distances_topk > cutoff] = 0

    padding_num = num_nn - D.shape[1] if num_nn > D.shape[1] else 0
    ids_topk = np.pad(ids_topk, ((0, 0), (0, padding_num)), constant_values=0)
    return ids_topk


def pack(token=None, xyz=None, nuv=None, llm=None, topk=None, y=None, msa=None, pssm=None, pdb_id=None):
    return {
        'token': token,
        'xyz': xyz,
        'nuv': nuv,
        'y': y,
        'llm': llm,
        'topk': topk,
        'msa': msa,
        'pssm': pssm,
        'pdb_id': pdb_id,
    }


def load_esm_feature(name, sequence, opt):
    llm_path = os.path.join(opt.dir_opts['ESM_AF_dir'], name + '.rep_5120.npy')
    return np.load(llm_path)[1:-1]


def load_msa_af_feature(name, sequence, opt):
    msa_af_path = os.path.join(opt.dir_opts['ESM_AF_dir'], name + 'msa_first_row.npy')
    return np.load(msa_af_path)


def load_esm_seq(name, opt):
    fasta_path = os.path.join(opt.dir_opts['ESM_AF_dir'], name + '.fasta')
    with open(fasta_path, 'r') as pid:
        lines = pid.readlines()
    return lines[1].strip()


def load_pssm_feature(name, sequence, opt):
    pssm_path = os.path.join(opt.dir_opts['pssm_dir'], f"{name}.pssm")
    if not os.path.exists(pssm_path):
        print(f"PSSM file missing for {name}: {pssm_path}")
        return np.zeros((len(sequence), opt.pssm_dim), dtype=np.float32)

    pssm_rows = []
    pssm_seq = []
    with open(pssm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 22 or not parts[0].isdigit() or len(parts[1]) != 1:
                continue
            try:
                scores = [float(x) for x in parts[2:22]]
            except ValueError:
                continue
            pssm_seq.append(parts[1])
            pssm_rows.append(scores)

    if len(pssm_rows) == 0:
        print(f"No valid PSSM rows found for {name}")
        return np.zeros((len(sequence), opt.pssm_dim), dtype=np.float32)

    pssm = np.asarray(pssm_rows, dtype=np.float32)
    seq_from_pssm = ''.join(pssm_seq)
    if len(seq_from_pssm) != len(sequence):
        print(f"PSSM length mismatch for {name}: seq={len(sequence)}, pssm={len(seq_from_pssm)}")
        return np.zeros((len(sequence), opt.pssm_dim), dtype=np.float32)
    if seq_from_pssm != sequence:
        print(f"PSSM sequence mismatch for {name}")
        return np.zeros((len(sequence), opt.pssm_dim), dtype=np.float32)
    return pssm


def load_hhm_feature(name, sequence, opt):
    if not opt.use_:
        return np.zeros((len(sequence), opt.hhm_dim), dtype=np.float32)

    if len(name) > 5 and name[5].islower():
        name = name[0:6] + name[5]

    hhm_path = os.path.join(opt.dir_opts['hhm_dir'], f"{name}.hhm")
    if not os.path.exists(hhm_path):
        print(f"HHM file missing for {name}: {hhm_path}")
        return np.zeros((len(sequence), opt.hhm_dim), dtype=np.float32)

    with open(hhm_path, 'r') as f:
        text = f.readlines()

    hhm_begin_line = 0
    hhm_end_line = 0
    for i in range(len(text)):
        if '#' in text[i]:
            hhm_begin_line = i + 5
        elif '//' in text[i]:
            hhm_end_line = i

    hhm = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
    axis_x = 0
    for i in range(hhm_begin_line, hhm_end_line, 3):
        line1 = text[i].split()[2:-1]
        line2 = text[i + 1].split()
        axis_y = 0
        for j in line1:
            hhm[axis_x][axis_y] = 9999 / 10000.0 if j == '*' else float(j) / 10000.0
            axis_y += 1
        for j in line2:
            hhm[axis_x][axis_y] = 9999 / 10000.0 if j == '*' else float(j) / 10000.0
            axis_y += 1
        axis_x += 1
    hhm = (hhm - np.min(hhm)) / (np.max(hhm) - np.min(hhm))
    return hhm.astype(np.float32)


def compute_pdb(protein, opt):
    name, sequence, label = protein[0], protein[1], protein[2]
    if opt.data_source == 'graphbind' and opt.dynamic_mode == 'static':
        if name[5].islower():
            name = name[0:6] + name[5]
    label = [int(i) for i in label]

    if opt.dynamic_mode == 'dynamic':
        token_, xyz_, nuv_, fasta_ = read_dynamic(name, opt.n_cluster, opt.ensemble, opt.dir_opts['ensemble_dir'], opt)
        if fasta_ == '':
            print(f"Dynamic pdb reading failed for {name}.")
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
    elif opt.dynamic_mode == 'static':
        token_, xyz_, nuv_, fasta_, label, sequence = read_PDB_compatible(opt.dir_opts['PDB_dir'], name + '.pdb', label)
    else:
        print(f"Invalid dynamic_mode: {opt.dynamic_mode}")
        return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')

    if opt.use_llm:
        try:
            if len(name) > 6:
                llm_ = load_esm_feature(name[:-1], sequence, opt)
            if len(name) == 6:
                llm_ = load_esm_feature(name, sequence, opt)
                fasta_llm = load_esm_seq(name, opt)
            else:
                print(name + '*' * 10)
                exit()
        except Exception as e:
            print(f"ESM feature extraction failed for {name}: {e}")
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
        if fasta_ != fasta_llm:
            print('Sequence string not match for llm of ID ' + name)
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
        if len(fasta_) != len(llm_):
            print('Sequence length not match for llm of ID ' + name)
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
    else:
        llm_ = np.zeros((len(sequence), opt.llm_dim), dtype=np.float32)

    if opt.use_msa:
        try:
            if len(name) > 6:
                msa_ = load_msa_af_feature(name[:-1], sequence, opt)
            if len(name) == 6:
                msa_ = load_msa_af_feature(name, sequence, opt)
            else:
                print(name + '*' * 10)
                exit()
        except Exception as e:
            print(f"AF feature extraction failed for {name}: {e}")
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
    else:
        msa_ = np.zeros((len(sequence), opt.llm_dim), dtype=np.float32)

    if opt.use_pssm:
        try:
            if len(name) > 6:
                pssm_ = load_pssm_feature(name[:-1], sequence, opt)
            if len(name) == 6:
                pssm_ = load_pssm_feature(name, sequence, opt)
            else:
                print(name + '*' * 10)
                exit()
        except Exception as e:
            print(f"pssm feature extraction failed for {name}: {e}")
            return pack(inttensor([]), tensor([]), tensor([]), tensor([]), tensor([]), inttensor([]), tensor([]), tensor(), '')
    else:
        pssm_ = np.zeros((len(sequence), opt.pssm_dim), dtype=np.float32)

    if opt.dynamic_mode == 'static':
        topk = extract_topology(xyz_)
    elif opt.dynamic_mode == 'dynamic':
        topk = []
        for k in range(xyz_.shape[0]):
            topk.append(extract_topology(xyz_[k]))
        topk = np.array(topk)

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


def load_pdb(proteins, opt, parallelize=False):
    print('Loading pdbs, paralleling: ', parallelize)
    if parallelize is False:
        pdbs_including_wrong = [compute_pdb(protein, opt) for protein in tqdm.tqdm(proteins)]
    else:
        n_jobs = cpu_count() // 2
        print('Using {} CPUs for parallel pdb loading.'.format(n_jobs))
        pdbs_including_wrong = Parallel(n_jobs=n_jobs, verbose=False, timeout=None)(
            delayed(compute_pdb)(protein, opt) for i, protein in enumerate(proteins)
        )

    pdbs = []
    for item in pdbs_including_wrong:
        if len(item['token']) >= 1:
            pdbs.append(item)
    return pdbs


def collate_fn_static(batch):
    following_batch = ['xyz']
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ['llm', 'msa'] and batch[0][key].numel() > 0:
            meta.update({key: torch.concat([d[key] for d in batch])})
        elif key in ['y', 'token', 'xyz', 'nuv', 'topk']:
            meta.update({key: torch.concat([d[key] for d in batch])})
            if key in following_batch:
                meta.update({key + '_batch': torch.concat(
                    [torch.ones(d[key].shape[0], dtype=torch.int64) * i for i, d in enumerate(batch)])})
        elif key == 'pdb_id':
            meta.update({key: [d[key] for d in batch]})
    return meta


def collate_multiconf(batch):
    meta = {}
    first_sample = batch[0]
    keys = first_sample.keys()
    sample_infos = []
    total_residues_all_confs = 0

    for sample in batch:
        k_current = sample['xyz'].shape[0]
        n_residues = sample['xyz'].shape[1]
        sample_infos.append({
            'k': k_current,
            'n': n_residues,
            'total_conformation_residues': k_current * n_residues,
        })
        total_residues_all_confs += k_current * n_residues

    adjustment_map = build_adjustment_map(sample_infos, total_residues_all_confs)

    for key in keys:
        if key in ['llm', 'msa', 'pssm'] and batch[0][key].numel() > 0:
            feat_tensors = []
            for sample, info in zip(batch, sample_infos):
                feat_repeated = sample[key].repeat(info['k'], 1)
                feat_tensors.append(feat_repeated)
            meta[key] = torch.cat(feat_tensors, dim=0)
        elif key == 'y':
            meta[key] = torch.cat([sample[key] for sample in batch], dim=0)
        elif key == 'token':
            token_tensors = []
            for sample, info in zip(batch, sample_infos):
                token_tensors.append(sample[key].repeat(info['k']))
            meta[key] = torch.cat(token_tensors, dim=0)
        elif key in ['xyz', 'nuv', 'topk']:
            concatenated_tensors = []
            for sample in batch:
                k, n = sample[key].shape[0], sample[key].shape[1]
                concatenated_tensors.append(sample[key].reshape(k * n, *sample[key].shape[2:]))
            meta[key] = torch.cat(concatenated_tensors, dim=0)
        elif key == 'pdb_id':
            meta.update({key: [d[key] for d in batch]})

    meta['xyz_batch'] = torch.zeros(total_residues_all_confs, dtype=torch.int64)
    meta['amino_acid_batch'] = torch.zeros(total_residues_all_confs, dtype=torch.int64)

    current_idx = 0
    amino_acid_counter = 0
    for i, info in enumerate(sample_infos):
        k, n = info['k'], info['n']
        meta['xyz_batch'][current_idx:current_idx + info['total_conformation_residues']] = i
        for amino_idx in range(n):
            start_idx = current_idx + amino_idx
            for conf_idx in range(k):
                global_idx = start_idx + conf_idx * n
                meta['amino_acid_batch'][global_idx] = amino_acid_counter
            amino_acid_counter += 1
        current_idx += info['total_conformation_residues']

    meta['cross_topk'] = build_cross_topk(sample_infos)
    if 'topk' in meta:
        meta['adjusted_topk'] = adjust_topk_fast(meta['topk'], adjustment_map)
    return meta


def build_cross_topk(sample_infos):
    total_residues_all_confs = sum(info['total_conformation_residues'] for info in sample_infos)
    max_k = max(info['k'] for info in sample_infos)
    cross_topk = torch.zeros(total_residues_all_confs, max_k, dtype=torch.long)
    current_idx = 0

    for info in sample_infos:
        k, n = info['k'], info['n']
        indices = torch.arange(current_idx, current_idx + k * n).reshape(k, n)
        for residue_pos in range(n):
            residue_conf_indices = indices[:, residue_pos]
            neighbor_indices_1based = residue_conf_indices + 1
            for conf_idx, global_idx in enumerate(residue_conf_indices):
                cross_topk[global_idx, :k] = neighbor_indices_1based
        current_idx += k * n
    return cross_topk


def build_adjustment_map(sample_infos, total_residues):
    adjustment_map = torch.zeros(total_residues, dtype=torch.long)
    current_idx = 0
    for info in sample_infos:
        k, n = info['k'], info['n']
        for conf_idx in range(k):
            conf_start = current_idx + conf_idx * n
            for local_pos in range(n):
                adjustment_map[conf_start + local_pos] = conf_start
        current_idx += info['total_conformation_residues']
    return adjustment_map


def adjust_topk_fast(topk, adjustment_map):
    total_residues, num_nn = topk.shape
    adjusted_topk = torch.zeros((total_residues + 1, num_nn), dtype=topk.dtype)
    adjusted_topk[0] = 1
    valid_mask = topk > 0
    adjustment_expanded = adjustment_map.unsqueeze(1).expand(-1, num_nn)
    adjusted_neighbors = torch.where(valid_mask, topk + adjustment_expanded, topk)
    adjusted_topk[1:] = adjusted_neighbors
    return adjusted_topk


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = self.CreateDataset()

        if opt.dynamic_mode == 'dynamic':
            collate_fn = collate_multiconf
        elif opt.dynamic_mode == 'static':
            collate_fn = collate_fn_static

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True if opt.subset == 'train' else False,
            collate_fn=collate_fn,
        )

    def CreateDataset(self):
        if self.opt.data_source == 'dyprol':
            if self.opt.ligand == 'RNA':
                train_file = str(DATASETS_DIR / 'DyProL' / 'RNA' / 'RNA-932_Train.txt')
            elif self.opt.ligand == 'DNA':
                train_file = str(DATASETS_DIR / 'DyProL' / 'DNA' / 'DNA-1022_Train.txt')

            if self.opt.ligand == 'RNA':
                test_file = str(DATASETS_DIR / 'DyProL' / 'RNA' / 'RNA-234_Test.txt')
            elif self.opt.ligand == 'DNA':
                test_file = str(DATASETS_DIR / 'DyProL' / 'DNA' / 'DNA-256_Test.txt')
        elif self.opt.data_source == 'graphbind':
            if self.opt.ligand == 'RNA':
                train_file = str(DATASETS_DIR / 'GraphBind' / 'RNA' / 'RNA-495_Train.txt')
            elif self.opt.ligand == 'DNA':
                train_file = str(DATASETS_DIR / 'GraphBind' / 'DNA' / 'DNA-573_Train.txt')

            if self.opt.ligand == 'RNA':
                test_file = str(DATASETS_DIR / 'GraphBind' / 'RNA' / 'RNA-117_Test.txt')
            elif self.opt.ligand == 'DNA':
                test_file = str(DATASETS_DIR / 'GraphBind' / 'DNA' / 'DNA-129_Test.txt')

        if self.opt.subset == 'train' or self.opt.subset == 'val':
            with open(train_file, 'r') as pid:
                train_text = pid.readlines()
            samples = []
            step = 4 if self.opt.data_source == 'graphbind' else 3
            for i in range(0, len(train_text), step):
                query_id = train_text[i].strip()[1:]
                query_seq = train_text[i + 1].strip()
                query_anno = train_text[i + 2].strip()
                samples.append([query_id, query_seq, query_anno])

        if self.opt.subset == 'test':
            with open(test_file, 'r') as pid:
                train_text = pid.readlines()
            samples = []
            for i in range(0, len(train_text), 3):
                query_id = train_text[i].strip()[1:]
                query_seq = train_text[i + 1].strip()
                query_anno = train_text[i + 2].strip()
                samples.append([query_id, query_seq, query_anno])

        random.seed(self.opt.seed)
        random.shuffle(samples)
        if self.opt.subset == 'train':
            samples = samples[int(len(samples) * 0.1):]
        if self.opt.subset == 'val':
            samples = samples[:int(len(samples) * 0.1)]
        samples = [samples[0], samples[1]]  # debug
        loaded_pdbs = load_pdb(samples, self.opt, parallelize=False)
        print(self.opt.subset, 'Total: {}'.format(len(samples)), 'pLDDT satisfied: {}'.format(len(loaded_pdbs)))
        return loaded_pdbs

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= len(self.dataset):
                break
            yield data
