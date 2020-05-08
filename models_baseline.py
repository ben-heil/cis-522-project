import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        outputs = self.linear(x)
        return outputs

    def train_forward(self, x, s, y, dummy_w=None):
        x = x.view(-1, self.input_dim)
        output = self.linear(x)

        loss = None
        if dummy_w is not None:
            loss = self.loss_fn(dummy_w * output, y)
        else:
            loss = self.loss_fn(output, y)

        num_correct = accuracy_score(output.max(1)[1].cpu().numpy(), y.squeeze().cpu().numpy(), normalize=False)
        return loss, num_correct

    def eval_forward(self, x, s):
        x = x.view(-1, self.input_dim)
        output = self.linear(x)
        return output
    
class CNN(nn.Module):
  def __init__(self, num_classes): 
    super(CNN, self).__init__() 
    self.features = nn.Sequential(nn.Conv2d(6, 16, kernel_size = 2, stride = 2, padding = 2),
                                  nn.ReLU(inplace = True),
                                  nn.MaxPool2d(kernel_size = 2, stride = 2),
                                  nn.Conv2d(16, 32, kernel_size = 2, stride = 2, padding = 2),
                                  nn.ReLU(inplace = True),
                                  nn.MaxPool2d(kernel_size = 2, stride = 2))
    self.classifier = nn.Sequential(nn.Linear(34848, 512),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(512, 256),
                                    nn.Linear(256, 128),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(128, num_classes))
    self.loss_fn = nn.CrossEntropyLoss()
    
  def forward(self, x): 
    x = self.features(x)
    x = x.view(x.size(0),-1)
    out = self.classifier(x)
    return out 
  
  def train_forward(self, x, s, y, dummy_w=None):
    x = self.features(x)
    x = x.view(x.size(0),-1)
    output = self.classifier(x)

    loss = None
    if dummy_w is not None:
        loss = self.loss_fn(dummy_w * output, y)
    else:
        loss = self.loss_fn(output, y)

    num_correct = accuracy_score(output.max(1)[1].cpu().numpy(), y.squeeze().cpu().numpy(), normalize=False)
    return loss, num_correct

  def eval_forward(self, x, s):
    x = self.features(x)
    x = x.view(x.size(0),-1)
    out = self.classifier(x)
    return out