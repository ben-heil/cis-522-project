'''
This file contains the functions to train models with invariant risk minimization
standard neural network trainnig (empirical risk minimization)
'''

import argparse
import os
import time
from typing import List
import json
# import logging
import sagemaker_containers
import sys


import numpy as np
import sklearn as skl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import datasets
from datasets import RecursionDataset
from models import ModelAndLoss, DenseNet


def compute_irm_penalty(loss, dummy_w):
    '''Calculate the invariance penalty for the classifier. This penalty is the norm of the
    gradient of the loss function multiplied by a dummy classifier with the value 1. This penalty
    constrains the model to perform well across studies. A more detailed explanation on why the
    dummy classifier is used can be found in section 3.1 of https://arxiv.org/abs/1907.02893
    '''
    dummy_grad = abs(grad(loss, dummy_w, create_graph=True)[0])

    return dummy_grad


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


def save_model(model, model_dir):

    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


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
    is_distributed = len(args.hosts) > 1 and args.backend is not None

    use_cuda = args.num_gpus > 0

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend,
                                rank=host_rank, world_size=world_size)

    # train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    # test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    model = net.to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    dummy_w = None

    batches = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_count = 0
        for batch in train_loader:
            image, cell_type, labels = batch
            image = image.float().to(device)
            labels = labels.to(device)
            cell_type = cell_type.to(
                device).float().view(-1, cell_type.size(-1))
            train_count += len(labels)

            optimizer.zero_grad()
            loss, acc = net.train_forward(image, cell_type, labels, dummy_w)

            train_correct += acc

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

    save_model(model, args.model_dir)

    return net


if __name__ == '__main__':
    # args = num_epochs, loss_scaling_factor

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])

    parser.add_argument(
        'data_dir', help='The path to the root of the data directory (called rxrx1 by default)')
    parser.add_argument('--num_epochs', default=100,
                        help='The number of epochs to train')
    parser.add_argument('--loss_scaling_factor', default=1,
                        help='The factor the loss is multiplied by before being added to the IRM '
                        'penalty. A larger factor emphasizes classification accuracy over '
                        'consistency across environments.')
    args = parser.parse_args()

    # data/rxrx1/rxrx1.csv

    train_dir = os.path.join(args.data_dir, 'images', 'train')

    metadata_df = datasets.load_metadata_df(
        os.path.join(args.data_dir, 'rxrx1.csv'))

    sirnas = metadata_df['sirna'].unique()
    sirna_encoder = skl.preprocessing.LabelEncoder()
    sirna_encoder.fit(sirnas)

    net = ModelAndLoss(len(sirnas)).to('cuda')
    #net = DenseNet(len(sirnas)).to('cuda')

    dataset1 = RecursionDataset(os.path.join(
        args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'HEPG2')
    dataset2 = RecursionDataset(os.path.join(
        args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'U2OS')
    val_dataset = RecursionDataset(os.path.join(
        args.data_dir, 'rxrx1.csv'), train_dir, sirna_encoder, 'train', 'HUVEC')

    loader1 = DataLoader(dataset1, batch_size=16, shuffle=True)
    loader2 = DataLoader(dataset2, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    writer = SummaryWriter('logs/irm{}'.format(time.time()))
    loaders = [loader1, loader2]
    train_erm(net, loader1, val_loader, writer, args)
    #train_irm(net, loaders, val_loader, writer, args)
