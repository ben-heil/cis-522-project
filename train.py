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
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datasets
from datasets import RecursionDataset
from models import ModelAndLoss, DenseNet, MultitaskNet


def compute_irm_penalty(loss, dummy_w):
    '''Calculate the invariance penalty for the classifier. This penalty is the norm of the
    gradient of the loss function multiplied by a dummy classifier with the value 1. This penalty
    constrains the model to perform well across studies. A more detailed explanation on why the
    dummy classifier is used can be found in section 3.1 of https://arxiv.org/abs/1907.02893
    '''
    dummy_grad = abs(grad(loss, dummy_w, create_graph=True)[0])

    return dummy_grad


def train_multitask(net: nn.Module, train_loaders: List[DataLoader], val_loader: DataLoader,
                    writer: SummaryWriter, args: argparse.Namespace):
    '''Train the given multitask network using.

    Arguments
    ---------
    net:
        The network to train
    train_loaders:
        A list containing a DataLoader for each cell type
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
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    batches = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_penalty = 0
        train_raw_loss = 0
        train_count = 0
        for env_loader in train_loaders:
            for batch in env_loader:
                image, cell_type, labels = batch
                image = image.float().to(device)
                labels = labels.to(device)
                cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))
                train_count += len(labels)

                optimizer.zero_grad()
                loss, acc = net.train_forward(image, cell_type, labels)
                train_correct += acc

                # Calculate the gradient of the combined loss function
                train_loss += loss.item()
                loss.backward(retain_graph=False)
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
                val_cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))

                val_count += len(labels)

                with torch.no_grad():
                    loss, acc = net.train_forward(
                        val_images, val_cell_type, val_labels)
                    val_loss += loss.item()
                    val_correct += acc

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    return net


def train_irm(net: nn.Module, train_loaders: List[DataLoader], val_loader: DataLoader, writer: SummaryWriter,
              args: argparse.Namespace):
    '''Train the given network using invariant risk minimization. This code is based on my
    implementation of IRM in https://github.com/ben-heil/whistl/, and by extension the original
    implementation by Arjovsky et al. 2019

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
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0])).to(device)

    batches = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_penalty = 0
        train_raw_loss = 0
        train_count = 0
        for env_loader in train_loaders:
            for batch in env_loader:
                image, cell_type, labels = batch
                image = image.float().to(device)
                labels = labels.to(device)
                cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))
                train_count += len(labels)

                optimizer.zero_grad()
                loss, acc = net.train_forward(
                    image, cell_type, labels, dummy_w)
                train_raw_loss += loss.item()
                train_correct += acc

                # This penalty is the norm of the gradient of 1 * the loss function.
                # The penalty helps keep the model from ignoring one study to the benefit
                # of the others, and the theoretical basis can be found in the Invariant
                # Risk Minimization paper
                penalty = compute_irm_penalty(loss, dummy_w)
                train_penalty += penalty.item()

                # Calculate the gradient of the combined loss function
                combined_loss = args.loss_scaling_factor * loss + penalty
                train_loss += combined_loss.item()
                combined_loss.backward(retain_graph=False)
                optimizer.step()

                if batches % 100 == 0:
                    train_loss = train_loss / train_count
                    train_raw_loss = train_raw_loss / train_count
                    train_acc = train_correct / train_count

                    writer.add_scalar('Loss/train', train_loss, batches)
                    writer.add_scalar('Raw_Loss/train',
                                      train_raw_loss, batches)
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
                val_cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))

                val_count += len(labels)

                with torch.no_grad():
                    loss, acc = net.train_forward(
                        val_images, val_cell_type, val_labels)
                    val_loss += loss.item()
                    val_correct += acc

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    return net


def train_erm(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              writer: SummaryWriter, args: argparse.Namespace):
    '''Train the given network using invariant risk minimization. This code is based on my
    implementation of IRM in https://github.com/ben-heil/whistl/, and by extension the original
    implementation by Arjovsky et al. 2019
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

    # dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0])).to(device) # dummy = 1
    dummy_w = None

    batches = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_count = 0
        for env_loader in train_loader:
            for batch in env_loader:
                image, cell_type, labels = batch
                image = image.float().to(device)
                labels = labels.to(device)
                cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))
                train_count += len(labels)

                optimizer.zero_grad()
                loss, acc = net.train_forward(
                    image, cell_type, labels, dummy_w)

                train_correct += acc

                train_loss += loss.item()
                loss.backward(retain_graph=False)
                optimizer.step()

                if batches % 100 == 0:
                    train_loss = train_loss / train_count
                    train_acc = train_correct / train_count

                    writer.add_scalar('Loss/train', train_loss, batches)
                    writer.add_scalar('Acc/train', train_acc, batches)

                    print("Epoch : %d, Batches Num : %d, train accuracy : %f" %
                          (epoch, batches, train_acc))

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
                val_cell_type = cell_type.to(
                    device).float().view(-1, cell_type.size(-1))

                val_count += len(labels)

                with torch.no_grad():
                    loss, acc = net.train_forward(
                        val_images, val_cell_type, val_labels)
                    val_loss += loss.item()
                    val_correct += acc

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    return net


def get_datasets(args: argparse.Namespace,
                 cell_type: str,
                 sirna_encoder: skl.preprocessing.LabelEncoder
                 ):
    '''Generate train and val RecursionDataset objects for a given cell type'''
    train_dir = os.path.join(args.data_dir, 'images', 'train')
    dataset = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'),
                               train_dir,
                               sirna_encoder,
                               'train',
                               cell_type
                               )
    data_len = len(dataset)
    train_data, val_data = torch.utils.data.random_split(dataset, (data_len // 10,
                                                                   data_len - data_len // 10
                                                                   )
                                                         )

    return train_data, val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The path to the root of the data directory '
                                         '(called rxrx1 by default)')
    parser.add_argument('--num_epochs', default=100,
                        help='The number of epochs to train')
    parser.add_argument('--loss_scaling_factor', default=1,
                        help='The factor the loss is multiplied by before being added to the IRM '
                        'penalty. A larger factor emphasizes classification accuracy over '
                        'consistency across environments.')
    args = parser.parse_args()

    # Create sirna encoder
    metadata_df = datasets.load_metadata_df(
        os.path.join(args.data_dir, 'rxrx1.csv'))

    sirnas = metadata_df['sirna'].unique()
    sirna_encoder = skl.preprocessing.LabelEncoder()
    sirna_encoder.fit(sirnas)

    HEPG2_train_data, HEPG2_val_data = get_datasets(
        args, 'HEPG2', sirna_encoder)
    HUVEC_train_data, HUVEC_val_data = get_datasets(
        args, 'HUVEC', sirna_encoder)
    RPE_train_data, RPE_val_data = get_datasets(args, 'RPE', sirna_encoder)
    combined_train_data = ConcatDataset(
        [HEPG2_train_data, HUVEC_train_data, RPE_train_data])
    val_data = ConcatDataset([HEPG2_val_data, HUVEC_val_data, RPE_val_data])

    HEPG2_train_loader = DataLoader(
        HEPG2_train_data, batch_size=16, shuffle=True)
    HUVEC_train_loader = DataLoader(
        HUVEC_train_data, batch_size=16, shuffle=True)
    RPE_train_loader = DataLoader(RPE_train_data, batch_size=16, shuffle=True)

    subset_indices = list(range(0, len(combined_train_data), 1000))

    combined_train_loader = DataLoader(
        combined_train_data, batch_size=16, shuffle=False, sampler=SubsetRandomSampler(subset_indices))
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    # Create test set
    train_dir = os.path.join(args.data_dir, 'images', 'train')
    U2OS_data = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'),
                                 train_dir,
                                 sirna_encoder,
                                 'train',
                                 'U2OS'
                                 )
    U2OS_loader = DataLoader(U2OS_data, batch_size=2, shuffle=False)

    # Initialize netork
    #net = ModelAndLoss(len(sirnas)).to('cuda')
    net = DenseNet(len(sirnas)).to('cuda')
    # net = MultitaskNet(len(sirnas)).to('cuda')

    writer = SummaryWriter('logs/erm{}'.format(time.time()))
    loaders = [HEPG2_train_loader, HUVEC_train_loader, RPE_train_loader]
    train_erm(net, combined_train_loader, val_loader, writer, args)
    writer = SummaryWriter('logs/irm{}'.format(time.time()))
    #train_irm(net, loaders, val_loader, writer, args)
    writer = SummaryWriter('logs/multitask_{}'.format(time.time()))
    # train_multitask(net, loaders, val_loader, writer, args)

    # train_erm(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    #           writer: SummaryWriter, args: argparse.Namespace)
