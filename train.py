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


def train_irm_load(net: nn.Module, train_loaders: List[DataLoader], val_loader: DataLoader, writer: SummaryWriter,
                   args: argparse.Namespace, optimizer: optim.Adam):
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


def save_checkpoint(model, optimizer, batch_num, base_name):
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_name = "saved_models/{}_{}.pth".format(base_name, batch_num)
    torch.save(checkpoint, save_name)


def train_erm_load_optimizer(net: nn.Module, train_loader: DataLoader, val_loader: DataLoader, writer: SummaryWriter, args: argparse.Namespace, optimizer: optim.Adam):
    '''adds optimizer argument that is loaded prior'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # optimizer = optim.Adam(net.parameters(), lr=1e-5)
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

                writer.add_scalar('Loss/train', train_loss, batches)
                writer.add_scalar('Acc/train', train_acc, batches)

                print("Epoch : %d, Batches : %d, train accuracy : %f" %
                      (epoch, batches, train_acc))

            if batches % 5000 == 0:
                save_checkpoint(net, optimizer, batches,
                                "{}_continued".format(args.checkpoint_name))

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

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    save_checkpoint(net, optimizer, batches,
                    "{}_continued_final".format(args.checkpoint_name))
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

                writer.add_scalar('Loss/train', train_loss, batches)
                writer.add_scalar('Acc/train', train_acc, batches)

                print("Epoch : %d, Batches : %d, train accuracy : %f" %
                      (epoch, batches, train_acc))

            if batches % 5000 == 0:
                save_checkpoint(net, optimizer, batches, args.checkpoint_name)

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

        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

    save_checkpoint(net, optimizer, batches,
                    "{}_final".format(args.checkpoint_name))
    return net


def get_datasets(args: argparse.Namespace,
                 cell_type: str,
                 sirna_encoder: skl.preprocessing.LabelEncoder,
                 sirnas_to_keep: List[int] = None,
                 ):
    '''Generate train and val RecursionDataset objects for a given cell type'''
    train_dir = os.path.join(args.data_dir, 'images', 'train')
    dataset = RecursionDataset(os.path.join(args.data_dir, 'rxrx1.csv'),
                               train_dir,
                               sirna_encoder,
                               'train',
                               cell_type,
                               sirnas_to_keep=sirnas_to_keep,
                               )
    data_len = len(dataset)
    train_data, val_data = torch.utils.data.random_split(dataset, (data_len - data_len // 10,
                                                                   data_len // 10
                                                                   )
                                                         )

    print(len(train_data), len(val_data))

    return train_data, val_data


def get_est_time():
    time_format = "%Y-%m-%d.%Hhours_%Mminutes_%Sseconds"
    current_time = datetime.now(timezone('US/Eastern'))
    return current_time.strftime(time_format)


def load_model_optimizer(model, optimizer, filename):
    if os.path.isfile(filename):
        print("Loading file {}".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("no file found at {}".format(filename))

    return model, optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The path to the root of the data directory '
                                         '(called rxrx1 by default)')
    parser.add_argument('model_type')
    parser.add_argument('train_type')
    parser.add_argument('checkpoint_name')
    parser.add_argument('--normalization', default = None,
                        help = 'Define normalization across a \'plate\', \'experiment\', or as none.'
                        '.csv with normalization values must be added to the data folder.')
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
    print("test", sirnas)
    sirnas = sirnas[:50]

    control_sirnas = ['s501309', 's501357', 's24587', 's2947', 's13580', 's1998', 's3887', 's502431', 's8645', 's501392', 's501323', 's12279', 's1174', 's7128', 's14729',
                      's15652', 's501295', 's501342', 'EMPTY', 's501307', 's194768', 's502432', 's4165', 's15788', 's18582', 's14484', 's5459', 's501351', 's501339', 'n337250', 's35651']
    control_sirnas_array = np.array(control_sirnas)
    sirnas = np.append(sirnas, control_sirnas_array)
    sirnas = np.unique(sirnas)
    

    sirna_encoder = skl.preprocessing.LabelEncoder()
    sirna_encoder.fit(sirnas)

    HEPG2_train_data, HEPG2_val_data = get_datasets(
        args, 'HEPG2', sirna_encoder, sirnas_to_keep=sirnas)
    HUVEC_train_data, HUVEC_val_data = get_datasets(
        args, 'HUVEC', sirna_encoder, sirnas_to_keep=sirnas)
    RPE_train_data, RPE_val_data = get_datasets(
        args, 'RPE', sirna_encoder, sirnas_to_keep=sirnas)
    combined_train_data = ConcatDataset(
        [HEPG2_train_data, HUVEC_train_data, RPE_train_data])
    val_data = ConcatDataset([HEPG2_val_data, HUVEC_val_data, RPE_val_data])

    # subset_indices = list(range(0, len(val_data), 100))

    HEPG2_train_loader = DataLoader(
        HEPG2_train_data, batch_size=16, shuffle=True)
    HUVEC_train_loader = DataLoader(
        HUVEC_train_data, batch_size=16, shuffle=True)
    RPE_train_loader = DataLoader(RPE_train_data, batch_size=16, shuffle=True)
    combined_train_loader = DataLoader(
        combined_train_data, batch_size=16, shuffle=True)

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

    loaders = [HEPG2_train_loader, HUVEC_train_loader, RPE_train_loader]
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
    else:
        print("invalid model type")

    if (args.train_type == 'erm'):
        print("training with erm")
        writer = SummaryWriter('logs/erm{}'.format(est_time))
        train_erm(net, combined_train_loader, val_loader, writer, args)
    elif (args.train_type == 'irm'):
        print("training with irm")
        writer = SummaryWriter('logs/irm{}'.format(est_time))

        train_irm(net, loaders, val_loader, writer, args)
    elif (args.train_type == 'multitask'):
        print("training with multitask")
        writer = SummaryWriter('logs/multitask_{}'.format(est_time))
        train_multitask(net, loaders, val_loader, writer, args)
    else:
        print("invalid train type")

    print("save final net")
    checkpoint = {
        'state_dict': net.state_dict(),
    }

    save_name = "saved_models/{}_finished.pth".format(args.checkpoint_name)
    torch.save(checkpoint, save_name)



    # Initialize netork
    # net = ModelAndLoss(len(sirnas)).to('cuda')
    # net = DenseNet(len(sirnas)).to('cuda')
    # net = MultitaskNet(len(sirnas)).to('cuda')

    # Unloaded
    # train_erm(net, combined_train_loader, val_loader, writer, args)
    # writer = SummaryWriter('logs/irm{}'.format(est_time))
    # train_irm(net, loaders, val_loader, writer, args)
    # writer = SummaryWriter('logs/multitask_{}'.format(est_time))
    # train_multitask(net, loaders, val_loader, writer, args)

    # load model changes

    # ERM Kaggle
    # print("latest")
    # writer = SummaryWriter('logs/erm{}'.format(est_time))
    # net = ModelAndLoss(len(sirnas)).to('cuda')
    # optimizer = optim.Adam(net.parameters(), lr=1e-5)
    # net_loaded, optimizer_loaded = load_model_optimizer(
    #     net, optimizer, 'saved_models/train_erm_kaggle_continued_24000.pth')
    # train_erm_load_optimizer(
    #     net_loaded, combined_train_loader, val_loader, writer, args, optimizer_loaded)

    # IRM Kaggle
    # print("kaggle irm continued")
    # net = ModelAndLoss(len(sirnas)).to('cuda')
    # writer = SummaryWriter('logs/irm{}'.format(est_time))
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # net_loaded, optimizer_loaded = load_model_optimizer(
    #     net, optimizer, 'saved_models/irm_kaggle_44000.pth')
    # train_irm_load(net_loaded, loaders, val_loader,
    #                writer, args, optimizer_loaded)
