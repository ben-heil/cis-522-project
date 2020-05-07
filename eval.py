'''
This file contains the functions to train models with invariant risk minimization
standard neural network trainnig (empirical risk minimization)
'''

import argparse
import os
import time
from pytz import timezone
from datetime import datetime
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
from sklearn.metrics import f1_score, accuracy_score


import datasets
from datasets import RecursionDataset
from models import ModelAndLoss, DenseNet, MultitaskNet
from models_baseline import LogisticRegression, CNN

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
    print("train loader length {}".format(len(train_loaders)))
    print("Val loader length {}".format(len(val_loader)))

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
                    print("Epoch : %d, Batches : %d, train accuracy : %f" %
                          (epoch, batches, train_acc))

                
                if batches % 5000 == 0:
                    save_checkpoint(net, optimizer, batches,
                                    args.checkpoint_name)
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
    print("train loader length {}".format(len(train_loaders)))
    print("Val loader length {}".format(len(val_loader)))

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
                    print("Epoch : %d, Batches : %d, train accuracy : %f" %
                          (epoch, batches, train_acc))

                if batches % 5000 == 0:
                    save_checkpoint(net, optimizer, batches,
                                    args.checkpoint_name)

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

        save_checkpoint(net, optimizer, batches,
                        "{}_final".format(args.checkpoint_name))

    return net


def eval_erm(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
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
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    # dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0])).to(device) # dummy = 1
    dummy_w = None

    batches = 0
    print("train loader length {}".format(len(train_loader)))
    print("Val loader length {}".format(len(val_loader)))
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
            loss.backward(retain_graph=False)  # modification here
            optimizer.step()

            if batches % 100 == 0:
                train_loss = train_loss / train_count
                train_acc = train_correct / train_count

                # writer.add_scalar('Loss/train', train_loss, batches)
                # writer.add_scalar('Acc/train', train_acc, batches)

                # print("Epoch : %d, Batches : %d, train accuracy : %f" %
                #       (epoch, batches, train_acc))

            # if batches % 5000 == 0:
            #     save_checkpoint(net, optimizer, batches, args.checkpoint_name)

            batches += 1

        val_loss = 0
        val_correct = 0
        val_count = 0
        # Speed up validation by telling torch not to worry about computing gradients

        with torch.no_grad():
            print("validation")
            for val_batch in val_loader:

                if (val_count % 100 == 0):
                    print("val batch {}".format(val_count))

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

        # if writer is not None:
        #     writer.add_scalar('Loss/val', val_loss, epoch)
        #     writer.add_scalar('Acc/val', val_acc, epoch)

    return net


def get_test_datasets(args: argparse.Namespace,
                 cell_type: str,
                 sirna_encoder: skl.preprocessing.LabelEncoder,
                 sirnas_to_keep: List[int] = None,
                 ):
    '''Generate test RecursionDataset objects for a given cell type'''
    test_dir = os.path.join(args.data_dir, 'images', 'test')
    dataset = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'),
                               test_dir,
                               sirna_encoder,
                               'test',
                               cell_type,
                               sirnas_to_keep=sirnas_to_keep,
                               )
    print(len(dataset))
    return dataset

def get_est_time():
    time_format = "%Y-%m-%d.%Hhours_%Mminutes_%Sseconds"
    current_time = datetime.now(timezone('US/Eastern'))
    return current_time.strftime(time_format)


def load_model(model, filename):
    if os.path.isfile(filename):
        print("Loading file {}".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("no file found at {}".format(filename))

    return model



def eval_baseline(net: nn.Module, test_loader: DataLoader,
              args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    val_loss = 0
    val_correct = 0
    val_count = 0
    predicted_labels = []
    actual_lables = []
    # Speed up validation by telling torch not to worry about computing gradients

    with torch.no_grad():
        print("evaluation")
        for test_batch in test_loader:

            if (val_count % 100 == 0):
                print("eval batch {}".format(val_count))

            images, cell_type, labels = test_batch
            val_images = images.float().to(device)
            val_labels = labels.to(device)
            val_cell_type = cell_type.to(
                device).float().view(-1, cell_type.size(-1))
            # print(len(actual_lables), len(val_labels))
            actual_lables.extend(val_labels.tolist())
            val_count += len(val_labels)

            pred = net.eval_forward(val_images, val_cell_type)
            pred = torch.argmax(pred.data, dim=1)
            # print(len(predicted_labels), len(pred))
            predicted_labels.extend(pred.tolist())

    # val_loss = val_loss / val_count
    # val_acc = val_correct / val_count

    # print('Loss/val', val_loss)
    # print('Acc/val', val_acc)
    
    # use predicted, actuals to compute metrics 
    # print(actual_lables)
    # print(predicted_labels)
    f1 = f1_score(actual_lables, predicted_labels, average='macro')
    acc = accuracy_score(predicted_labels, actual_lables)
    print('f1:', f1, '\tacc:', acc)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The path to the root of the data directory '
                                         '(called rxrx1 by default)')
    parser.add_argument('model_type')
    parser.add_argument('train_type')
    parser.add_argument('model_path')

    args = parser.parse_args()

    # Create sirna encoder
    metadata_df = datasets.load_metadata_df(
        os.path.join(args.data_dir, 'rxrx1.csv'))

    sirnas = metadata_df['sirna'].unique()
    sirnas = sirnas[:50]

    control_sirnas = ['s501309', 's501357', 's24587', 's2947', 's13580', 's1998', 's3887', 's502431', 's8645', 's501392', 's501323', 's12279', 's1174', 's7128', 's14729',
                      's15652', 's501295', 's501342', 'EMPTY', 's501307', 's194768', 's502432', 's4165', 's15788', 's18582', 's14484', 's5459', 's501351', 's501339', 'n337250', 's35651']
    control_sirnas_array = np.array(control_sirnas)
    sirnas = np.append(sirnas, control_sirnas_array)
    sirnas = np.unique(sirnas)
    
    sirna_encoder = skl.preprocessing.LabelEncoder()
    sirna_encoder.fit(sirnas)

    # Create Dataset by Cell Type
    HEPG2_test_data = get_test_datasets(args, 'HEPG2', sirna_encoder, sirnas_to_keep=sirnas)
    HUVEC_test_data = get_test_datasets(args, 'HUVEC', sirna_encoder, sirnas_to_keep=sirnas)
    RPE_test_data = get_test_datasets(args, 'RPE', sirna_encoder, sirnas_to_keep=sirnas)

    combined_test_data = ConcatDataset([HEPG2_test_data, HUVEC_test_data, RPE_test_data])

    # Crate Dataloader by Cell Type
    HEPG2_test_loader = DataLoader(HEPG2_test_data, batch_size=16, shuffle=True)
    HUVEC_test_loader = DataLoader(HUVEC_test_data, batch_size=16, shuffle=True)
    RPE_test_loader = DataLoader(RPE_test_data, batch_size=16, shuffle=True)
    
    combined_test_loader = DataLoader(combined_test_data, batch_size=16, shuffle=True)

    # List of 3 loaders
    loaders = [HEPG2_test_loader, HUVEC_test_loader, RPE_test_loader]

    # Create test set
    U2OS_test_data = get_test_datasets(args, 'U2OS', sirna_encoder, sirnas_to_keep=sirnas)
    U2OS_test_loader = DataLoader(U2OS_test_data, batch_size=16, shuffle=False)

    # combined_test_loader = U2OS_test_loader
    est_time = get_est_time()
    net = None

    if (args.model_type == "densenet"):
        print("you picked densenet")
        net = DenseNet(len(sirnas)).to('cuda')
    elif (args.model_type == "kaggle"):
        print("you picked kaggle")
        net = ModelAndLoss(len(sirnas)).to('cuda')
    elif(args.model_type == "multitask"):
        print("you picked multitask")
        net = MultitaskNet(len(sirnas)).to('cuda')
    elif(args.model_type == "lr"):
        print("you picked lr")
        net = LogisticRegression(512*512*6, len(sirnas)).to('cuda')
    elif(args.model_type == "cnn"):
        print("you picked cnn")
        net = CNN(len(sirnas)).to('cuda')
    else:
        print("invalid model type")

    # Load from Checkpoint
    net = load_model(net, args.model_path)

    if (args.train_type == 'erm'):
        print("eval erm")
        eval_erm(net, combined_test_loader, args)
    elif (args.train_type == 'irm'):
        print("eval irm")
        train_irm(net, combined_test_loader, args)
    elif (args.train_type == 'multitask'):
        print("eval multitask")
        train_multitask(net, combined_test_loader, args)
    elif (args.train_type == 'baseline'):
        print("eval baseline")
        eval_baseline(net, combined_test_loader, args)
    else:
        print("invalid train type")
