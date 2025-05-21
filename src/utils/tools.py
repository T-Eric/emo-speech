import os
import numpy as np
import torch
from .. import config
from sklearn.model_selection import StratifiedKFold


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def load_data(data_dir=config.PROCESSED_DATA_DIR):
    '''
    Load features and labels, then form their graphs.
    No need to separate data into training and validation sets, since we use K-fold cross-validation.
    TODO 可能还缺少一步正则化
    '''
    print('Loading features and labels...')
    ret = []
    actor_dir_name = 'Actor_'
    for i in range(24):
        dir_path = os.path.join(data_dir, actor_dir_name+str(i+1).zfill(2)+'/')
        for _, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.npz'):
                    data = np.load(os.path.join(dir_path, file))
                    ret.append([data['feature'], int(data['label'])])
    print(f'Done. Feature shape:{np.array(ret[0][0]).shape}')
    return ret, ret[0][0].shape[0], ret[0][0].shape[1]


def separate_data(data, seed, k=10):
    '''Use K-fold cross-validation to separate data into training and validation sets with randomness.'''
    labels = [item[1] for item in data]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    train_portions = []
    test_portions = []

    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        train_portions.append([data[i] for i in train_idx])
        test_portions.append([data[i] for i in test_idx])

    return train_portions, test_portions


def adj_builder(n, adj_num=1, self_connect=False):
    '''
    Build adjacency matrix for graph convolution.
    '''
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        if self_connect:
            A[i, i] = 1
        for j in range(1, adj_num+1):
            A[i, (i-j) % n] = 1
            A[i, (i+j) % n] = 1

    return A
