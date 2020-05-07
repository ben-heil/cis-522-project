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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


import datasets
from datasets import RecursionDataset
from models import ModelAndLoss, DenseNet, MultitaskNet
from models_baseline import LogisticRegression, CNN

from textwrap import wrap
import re
import itertools
import matplotlib
import matplotlib.pyplot as plt

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
    return dataset


def load_model(model, filename):
    if os.path.isfile(filename):
        print("Loading file {}".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("no file found at {}".format(filename))

    return model

def plot_confusion_matrix(correct_labels, predict_labels, labels, display_labels, path, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
  ''' 
  Parameters:
      correct_labels                  : These are your true classification categories.
      predict_labels                  : These are you predicted classification categories
      labels                          : This is a lit of labels which will be used to display the axix labels
      title='Confusion matrix'        : Title for your matrix
      tensor_name = 'MyFigure/image'  : Name for the output summay tensor

  Returns:
      summary: TensorFlow summary 

  Other itema to note:
      - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
      - Currently, some of the ticks dont line up due to rotations.
  '''
  cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
  if normalize:
      cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
      cm = np.nan_to_num(cm, copy=True)
      cm = cm.astype('int')

  np.set_printoptions(precision=2)
  ###fig, ax = matplotlib.figure.Figure()

  fig = matplotlib.pyplot.figure(figsize=(2, 2), dpi=320, facecolor='w', edgecolor='k')
  ax = fig.add_subplot(1, 1, 1)
  im = ax.imshow(cm, cmap='Oranges')

  classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in display_labels]
  classes = ['\n'.join(wrap(l, 40)) for l in classes]

  tick_marks = np.arange(len(classes))

  ax.set_xlabel('Predicted', fontsize=7)
  ax.set_xticks(tick_marks)
  c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  ax.set_ylabel('True Label', fontsize=7)
  ax.set_yticks(tick_marks)
  ax.set_yticklabels(classes, fontsize=4, va ='center')
  ax.yaxis.set_label_position('left')
  ax.yaxis.tick_left()

#   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#       ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
  fig.set_tight_layout(True)
  save_name = 'confusion_matrix/' + path[path.index('/')+1:path.index('.pth')] + ".png"
  matplotlib.pyplot.savefig(save_name)
  matplotlib.pyplot.show()

  return


def plot_matrix(correct_labels, predict_labels, path):
    label_domain = [i for i in range(0,74)]
    display_labels = [str(i) for i in range(0,74)]
    plot_confusion_matrix(correct_labels, predict_labels, label_domain, display_labels, path)


def eval_model(net: nn.Module, test_loader: DataLoader,
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
    plot_matrix(actual_lables, predicted_labels, args.model_path)
    print('f1:', f1, '\tacc:', acc)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='The path to the root of the data directory '
                                         '(called rxrx1 by default)')
    parser.add_argument('model_type')
    parser.add_argument('model_path')
    
    parser.add_argument('--sirna_selection', default="subset",
                        help='subset or full')
    parser.add_argument('--eval_set', default="holdout",
                        help='combined or holdout')


    args = parser.parse_args()

    # Create sirna encoder
    metadata_df = datasets.load_metadata_df(
        os.path.join(args.data_dir, 'rxrx1.csv'))


    if (args.sirna_selection == 'subset'):
        sirnas = metadata_df['sirna'].unique()
        sirnas = sirnas[:50]

        control_sirnas = ['s501309', 's501357', 's24587', 's2947', 's13580', 's1998', 's3887', 's502431', 's8645', 's501392', 's501323', 's12279', 's1174', 's7128', 's14729',
                        's15652', 's501295', 's501342', 'EMPTY', 's501307', 's194768', 's502432', 's4165', 's15788', 's18582', 's14484', 's5459', 's501351', 's501339', 'n337250', 's35651']
        control_sirnas_array = np.array(control_sirnas)
        sirnas = np.append(sirnas, control_sirnas_array)
        sirnas = np.unique(sirnas)
        print("Sirna selection: Subset\t\t\t\t\tNum Sirnas:", len(sirnas))
    elif (args.sirna_selection == 'full'):
        sirnas = metadata_df['sirna'].unique()
        print("Sirna selection: Full\t\t\t\t\tNum Sirnas:", len(sirnas))
    else:
        print("Sirna selection: INVALID")

    sirna_encoder = skl.preprocessing.LabelEncoder()
    labels = sirna_encoder.fit_transform(sirnas)

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

    if (args.eval_set == 'holdout'):
        print('Evaluation Selection: U2OS (holdout)\t\t\tSize:', len(U2OS_test_data))
        test_loader = U2OS_test_loader
    elif (args.eval_set == 'combined'):
        print('Evaluation Selection: HEPG2, HUVEC, RPE\t\t\tSize:', len(combined_test_data))
        test_loader = combined_test_loader
    else:
        print('Evaluation Selection: INVALID')

    combined_test_loader = U2OS_test_loader
    net = None

    if (args.model_type == "densenet"):
        print("Model: densenet")
        net = DenseNet(len(sirnas)).to('cuda')
    elif (args.model_type == "kaggle"):
        print("Model: kaggle")
        net = ModelAndLoss(len(sirnas)).to('cuda')
    elif(args.model_type == "multitask"):
        print("Model: multitask")
        net = MultitaskNet(len(sirnas)).to('cuda')
    elif(args.model_type == "lr"):
        print("Model: lr")
        net = LogisticRegression(512*512*6, len(sirnas)).to('cuda')
    elif(args.model_type == "cnn"):
        print("Model: cnn")
        net = CNN(len(sirnas)).to('cuda')
    else:
        print("invalid model type")

    # Load from Checkpoint
    print("Beginning evaluation")
    net = load_model(net, args.model_path)
    eval_model(net, test_loader, args)
