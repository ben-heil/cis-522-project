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
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        outputs = self.linear(x)
        return outputs
    
    
class CNN(nn.Module):
  def __init__(self, num_classes): 
    super(CNN, self).__init__() 
    self.features = nn.Sequential(nn.Conv2d(6, 16, kernel_size = 5, stride = 2, padding = 2),
                                  nn.ReLU(inplace = True),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2),
                                  nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 2),
                                  nn.ReLU(inplace = True),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2),
                                  nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 2),
                                  nn.ReLU(inplace = True),
                                  nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.classifier = nn.Sequential(nn.Linear(16384, 1024),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(512, 256),
                                    nn.Linear(256, 128),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(128, num_classes))
  
  def forward(self, x): 
    x = self.features(x)
    x = x.view(x.size(0),-1)
    out = self.classifier(x)
    return out 