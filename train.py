import time, os, sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = Path(__file__).resolve().parent

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from sklearn.metrics import roc_auc_score, average_precision_score

from Arguments import parser
from dir_opts import dir_opts
from dataloader.dataloader import DataLoader
from model import Main_model
from model import EarlyStopping

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable: ', trainable_num)

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

def train_batch(data, model):
    model.set_input(data)
    out = model.optimize_parameters()
    loss = model.loss
    return loss

def val_batch(data, model):
    model.set_input(data)
    loss, true, pre = model.test()
    return loss, true, pre

if __name__ == '__main__':
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    model = Main_model(args)
    get_parameter_number(model)
    args.checkpoints_dir = checkpoints_dir
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    opts = dir_opts(args, PROJECT_ROOT)
    setattr(args, 'dir_opts', opts)

    setattr(args, 'subset', 'train')
    dataset_train = DataLoader(args)
    setattr(args, 'subset', 'val')
    dataset_val = DataLoader(args)
    setattr(args, 'subset', 'test')
    dataset_test = DataLoader(args)

    early_stop = EarlyStopping(opt=args, path=os.path.join(args.checkpoints_dir, 'best.pth'))
    total_steps = 0

    auc_best = 0
    time1 = time.time()
    curve = []
    for epoch in range(100):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in tqdm(enumerate(dataset_train)):
            loss_ = train_batch(data, model)

#        loss_train = []; proba_train = []; true_train = []
#        for i, data in enumerate(dataset_train):
#            loss_, true_, proba_ = val_batch(data, model)
#            loss_train.append(loss_.detach().cpu().numpy())
#            proba_train.append(proba_.detach().cpu().numpy())
#            true_train.append(true_.detach().cpu().numpy())

        loss_val = []; proba_val = []; true_val = []
        for i, data in enumerate(dataset_val):
            loss_, true_, proba_ = val_batch(data, model)
            loss_val.append(loss_.detach().cpu().numpy())
            proba_val.append(proba_.detach().cpu().numpy())
            true_val.append(true_.detach().cpu().numpy())

        loss_test = []; proba_test = []; true_test = []
        for i, data in enumerate(dataset_test):
            loss_, true_, proba_ = val_batch(data, model)
            loss_test.append(loss_.detach().cpu().numpy())
            proba_test.append(proba_.detach().cpu().numpy())
            true_test.append(true_.detach().cpu().numpy())

#        loss_train = np.average(np.array(loss_train))
#        pre_train = np.vstack(proba_train)
#        true_train = np.hstack(true_train)
#        auc_train = roc_auc_score(true_train, pre_train[:,1])
#        auprc_train = average_precision_score(true_train, pre_train[:,1])

        pre_val = np.vstack(proba_val)
        true_val = np.hstack(true_val)
        auc_val = roc_auc_score(true_val, pre_val[:,1])
        auprc_val = average_precision_score(true_val, pre_val[:,1])

        pre_test = np.vstack(proba_test)
        true_test = np.hstack(true_test)
        auc_test = roc_auc_score(true_test, pre_test[:,1])
        auprc_test = average_precision_score(true_test, pre_test[:,1])

        printf(args, 'Epoch: ', str(epoch),
#               'Loss_train: ', str(loss_train)[0:5],
#               'AUPRC_train:', str(auprc_train)[0:6],
               'AUPRC_val:', str(auprc_val)[0:6],
               'AUPRC_test:', str(auprc_test)[0:6],
#                'AUC_train:', str(auc_train)[0:5],
               'AUC_val:', str(auc_val)[0:5],
               'AUC_test:', str(auc_test)[0:5],
                )
#        curve.append([loss_train, auprc_train, auprc_val, auprc_test, auc_train, auc_val, auc_test])
        curve.append([ auprc_val, auprc_test, auc_val, auc_test])
        early_stop(auc_val, model)
        if early_stop.early_stop == True:
            break
    curve = pd.DataFrame(curve)
    curve.to_csv(os.path.join(args.checkpoints_dir, 'curve.csv'))
