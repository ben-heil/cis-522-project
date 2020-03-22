'''
This file contains the functions to train models with invariant risk minimization
standard neural network trainnig (empirical risk minimization)
'''

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_irm(net: nn.Module, train_loaders: list, val_loader: DataLoader, writer: SummaryWriter):
    '''Train the given network using invariant risk minimization

    Arguments
    ---------
    net:
        The network to train
    train_loaders:
        A list containing a DataLoader for each environment
    val_loader:
        The dataloader containing the validation dataset
    writer:
        The SummaryWriter to write results to

    Returns
    -------
    net:
        The network after training is finished
    '''
    raise NotImplementedError

def train_erm(net: nn.Module, train_loader: DataLoader, writer: SummaryWriter):
    '''Train the given network using empirical risk minimization (standard training)

    Arguments
    ---------
    net:
        The network to train
    train_loader:
        The data containing all training data and labels
    val_loader:
        The dataloader containing the validation dataset
    writer:
        The SummaryWriter to write results to

    Returns
    -------
    net:
        The network after training is finished
    '''
    raise NotImplementedError


if __name__ == '__main__':
    raise NotImplementedError
