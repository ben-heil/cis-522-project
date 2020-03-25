'''
This file contains the functions to train models with invariant risk minimization
standard neural network trainnig (empirical risk minimization)
'''

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def compute_irm_penalty(loss, dummy_w):
    '''Calculate the invariance penalty for the classifier. This penalty is the norm of the
    gradient of the loss function multiplied by a dummy classifier with the value 1. This penalty
    constrains the model to perform well across studies. A more detailed explanation on why the
    dummy classifier is used can be found in section 3.1 of https://arxiv.org/abs/1907.02893
    '''
    dummy_grad = abs(grad(loss, dummy_w, create_graph=True)[0])

    return dummy_grad


def train_irm(net: nn.Module, train_loaders: list, val_loader: DataLoader, writer: SummaryWriter,
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
    # TODO fix optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0])).to(device)

    for epoch in tqdm(range(args.num_epochs)):
        train_correct = 0
        train_loss = 0
        train_penalty = 0
        train_raw_loss = 0
        train_count = 0
        for env_loader in train_loaders:
            for batch in env_loader:
                image, cell_type, _, labels = batch
                image = image.float().to(device)
                labels = labels.to(device).float()
                cell_type = cell_type.to(device)
                train_count += len(labels)

                loss, acc = net.train_forward(image, cell_type, labels)
                train_raw_loss += loss
                train_correct += acc * len(labels)

            # This penalty is the norm of the gradient of 1 * the loss function.
            # The penalty helps keep the model from ignoring one study to the benefit
            # of the others, and the theoretical basis can be found in the Invariant
            # Risk Minimization paper
            penalty = compute_irm_penalty(loss, dummy_w)
            train_penalty += penalty.item()

            optimizer.zero_grad()
            # Calculate the gradient of the combined loss function
            combined_loss = args.loss_scaling_factor * loss + penalty
            train_loss += combined_loss.item()
            combined_loss.backward(retain_graph=False)
            optimizer.step()

        val_loss = 0
        val_correct = 0
        val_count = 0
        # Speed up validation by telling torch not to worry about computing gradients
        with torch.no_grad():
            for val_batch in val_loader:
                images, cell_type, _, labels = val_batch
                val_images = images.float().to(device)
                val_labels = labels.to(device).float()
                val_cell_type = cell_type.to(device)
                val_count += len(labels)

                with torch.no_grad():
                    loss, acc = net.train_forward(val_images, val_cell_type, val_labels)
                    val_loss += loss.item()
                    val_correct += acc * len(labels)

        val_loss = val_loss / val_count
        val_acc = val_correct / val_count
        train_loss = train_loss / train_count
        train_acc = train_correct / train_count

        train_penalty = train_penalty.item()
        train_raw_loss = train_raw_loss.item()

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    return net

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
    # args = num_epochs, loss_scaling_factor

    parser = argparse.ArgumentParser()
