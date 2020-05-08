'''
This file contains the functions to train models with invariant risk minimization
standard neural network trainnig (empirical risk minimization)
'''

import argparse
import os
import time
from typing import List

import numpy as np
import sklearn as skl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datasets
from datasets import RecursionDataset
from models import ModelAndLoss, DenseNet
from models_baseline import LogisticRegression, CNN



def train_baseline(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              writer: SummaryWriter, args: argparse.Namespace):
    '''
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss() 

    batches = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_count = 0
        for batch in train_loader:
            image, cell_type, labels = batch
            image = image.float().to(device)
            labels = labels.to(device)
            cell_type = cell_type.to(device).float().view(-1, cell_type.size(-1))
            train_count += len(labels)         
            optimizer.zero_grad()
            output = net(image)

            loss = criterion(output, labels)
            _,pred = torch.max(output,1)
            train_correct += (torch.sum(pred == labels)).item()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if batches % 100 == 0:
                train_loss = train_loss / train_count
                train_acc = train_correct / train_count

                writer.add_scalar('Loss/train', train_loss, batches)
                writer.add_scalar('Acc/train', train_acc, batches)
            batches += 1

        val_loss = 0
        val_correct = 0
        val_count = 0
        # Speed up validation by telling torch not to worry about computing gradients
        with torch.no_grad():
            for val_batch in val_loader:
                images, cell_type, labels = val_batch
                val_images = images.float().to(device)
                val_labels = labels.to(device)
                val_cell_type = cell_type.to(device).float().view(-1, cell_type.size(-1))

                val_count += len(labels)

                with torch.no_grad():
                    output = net(image)
                    loss = criterion(output, labels)
                    _,pred = torch.max(output,1)
                    val_correct += (torch.sum(pred == labels)).item()
                    val_loss += loss.item()
                    

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    return net


if __name__ == '__main__':
    # args = num_epochs, loss_scaling_factor

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The path to the root of the data directory (called rxrx1 by default)')
    parser.add_argument('--num_epochs', default=100, help='The number of epochs to train')
    parser.add_argument('--loss_scaling_factor', default=1,
                        help='The factor the loss is multiplied by before being added to the IRM '
                        'penalty. A larger factor emphasizes classification accuracy over '
                        'consistency across environments.')
    args = parser.parse_args()

    # data/rxrx1/rxrx1.csv

    train_dir = os.path.join(args.data_dir, 'images', 'train')

    metadata_df = datasets.load_metadata_df(os.path.join(args.data_dir, 'rxrx1.csv'))

    sirnas = metadata_df['sirna'].unique()
    sirna_encoder = skl.preprocessing.LabelEncoder()
    sirna_encoder.fit(sirnas)

    net = LogisticRegression(512*512*6, len(sirnas))
    # net = CNN(len(sirnas))
    
    # print(len(sirnas))  ==> 1139
    dataset1 = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'HEPG2')
    dataset2 = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'U2OS')
    val_dataset = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'HUVEC')

    loader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
    loader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)


    writer = SummaryWriter('logs/baseline{}'.format(time.time()))
    loaders = [loader1, loader2]
    
    train_baseline(net, loader1, val_loader, writer, args)
