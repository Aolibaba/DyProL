import csv
import os
import random
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = Path(__file__).resolve().parent

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from Arguments import parser

warnings.filterwarnings('ignore', category=RuntimeWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable: ', trainable_num)


def printf(args, *content):
    message = ' '.join(str(item) for item in content)
    with open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+') as f_handler:
        print(message, file=f_handler)
    print(message)


def train_batch(data, model):
    model.set_input(data)
    return model.optimize_parameters()


def val_batch(data, model):
    model.set_input(data)
    loss, true, pre = model.test()
    return loss, true, pre


def compute_binary_metrics(true, proba):
    if len(np.unique(true)) < 2:
        return np.nan, np.nan
    return roc_auc_score(true, proba[:, 1]), average_precision_score(true, proba[:, 1])


def evaluate(dataset, model):
    losses = []
    probabilities = []
    labels = []
    for data in dataset:
        loss, true, proba = val_batch(data, model)
        losses.append(loss.detach().cpu().numpy())
        probabilities.append(proba.detach().cpu().numpy())
        labels.append(true.detach().cpu().numpy())

    if not probabilities:
        return np.nan, np.nan, np.nan

    proba = np.vstack(probabilities)
    true = np.hstack(labels)
    auc, auprc = compute_binary_metrics(true, proba)
    return np.average(np.array(losses)), auc, auprc


if __name__ == '__main__':
    args = parser.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from tqdm import tqdm

    from dataloader.dataloader import DataLoader
    from dir_opts import dir_opts
    from model import EarlyStopping, Main_model

    set_seed(args.seed)
    if args.checkpoints_dir is None:
        args.checkpoints_dir = os.path.join(
            'Checkpoints',
            'GraphBind',
            args.ligand,
        )

    checkpoints_dir = args.checkpoints_dir
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    opts = dir_opts(args, PROJECT_ROOT)
    setattr(args, 'dir_opts', opts)

    model = Main_model(args)
    get_parameter_number(model)
    args.checkpoints_dir = checkpoints_dir

    setattr(args, 'subset', 'train')
    dataset_train = DataLoader(args)
    setattr(args, 'subset', 'val')
    dataset_val = DataLoader(args)
    setattr(args, 'subset', 'test')
    dataset_test = DataLoader(args)

    early_stop = EarlyStopping(opt=args, path=os.path.join(args.checkpoints_dir, 'best.pth'))

    curve = []
    for epoch in range(100):
        for data in tqdm(dataset_train):
            train_batch(data, model)

        loss_val, auc_val, auprc_val = evaluate(dataset_val, model)
        loss_test, auc_test, auprc_test = evaluate(dataset_test, model)

        printf(args, 'Epoch: ', str(epoch),
               'Loss_val:', str(loss_val)[0:6],
               'AUPRC_val:', str(auprc_val)[0:6],
               'Loss_test:', str(loss_test)[0:6],
               'AUPRC_test:', str(auprc_test)[0:6],
               'AUC_val:', str(auc_val)[0:5],
               'AUC_test:', str(auc_test)[0:5],
                )
        curve.append([loss_val, auprc_val, loss_test, auprc_test, auc_val, auc_test])
        fitness = auc_val if not np.isnan(auc_val) else -np.inf
        early_stop(fitness, model)
        if early_stop.early_stop:
            break
    curve_path = os.path.join(args.checkpoints_dir, 'curve.csv')
    curve_columns = ['loss_val', 'auprc_val', 'loss_test', 'auprc_test', 'auc_val', 'auc_test']
    with open(curve_path, 'w', newline='') as curve_file:
        writer = csv.writer(curve_file)
        writer.writerow(curve_columns)
        writer.writerows(curve)
