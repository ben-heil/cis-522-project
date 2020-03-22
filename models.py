'''This file contains the model classes for use in predicting cell perturbations'''

import torch.nn as nn
import torch.nn.functional as F

class RecursionModel(nn.Module):
    '''The winning solution from the Recursion.AI kaggle competition'''
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
