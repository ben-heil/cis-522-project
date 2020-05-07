'''This file contains the model classes for use in predicting cell perturbations.
The code for Model, ModelAndLoss, and their subclasses are a modified version of
https://github.com/maciej-sypetkowski/kaggle-rcic-1st/blob/master/model.py
The code for DenseNet and MultitaskNet was written by our team to
match the API present in ModelAndLoss to allow uniform training
'''

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing


def one_hot_encode(labels, num_classes):
    '''Manually one hot encode labels because sklearn's encoder is tempermental
    and DenseCrossEntropy wants the labels to be one hot (or pseudolabeled)
    '''
    encoded = np.zeros((len(labels), num_classes))

    for index, label in enumerate(labels):
        encoded[index, int(label.cpu())] = 1
    return torch.FloatTensor(encoded).cuda()


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Set arguments to match winning defaults
        args = SimpleNamespace()
        args.backbone = 'mem-densenet161'
        args.concat_cell_type = True
        args.classes = num_classes
        args.embedding_size = 1024
        args.head_hidden = None
        args.bn_mom = 0.05

        kwargs = {}
        backbone = args.backbone
        if args.backbone.startswith('mem-'):
            kwargs['memory_efficient'] = True
            backbone = args.backbone[4:]

        if backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(6, channels, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True, **kwargs)
            self.features = pretrained_backbone.features
            self.features.conv0 = first_conv
            features_num = pretrained_backbone.classifier.in_features
        elif backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True, **kwargs)
            self.features = nn.Sequential(
                first_conv,
                pretrained_backbone.bn1,
                pretrained_backbone.relu,
                pretrained_backbone.maxpool,
                pretrained_backbone.layer1,
                pretrained_backbone.layer2,
                pretrained_backbone.layer3,
                pretrained_backbone.layer4,
            )
            features_num = pretrained_backbone.fc.in_features
        else:
            raise ValueError('wrong backbone')

        self.concat_cell_type = args.concat_cell_type
        self.classes = args.classes

        features_num = features_num + (4 if self.concat_cell_type else 0)

        self.neck = nn.Sequential(
            nn.BatchNorm1d(features_num),
            nn.Linear(features_num, args.embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(args.embedding_size),
            nn.Linear(args.embedding_size, args.embedding_size, bias=False),
            nn.BatchNorm1d(args.embedding_size),
        )
        self.arc_margin_product = ArcMarginProduct(args.embedding_size, args.classes)

        if args.head_hidden is None:
            self.head = nn.Linear(args.embedding_size, args.classes)
        else:
            self.head = []
            for input_size, output_size in zip([args.embedding_size] + args.head_hidden, args.head_hidden):
                self.head.extend([
                    nn.Linear(input_size, output_size, bias=False),
                    nn.BatchNorm1d(output_size),
                    nn.ReLU(),
                ])
            self.head.append(nn.Linear(args.head_hidden[-1], args.classes))
            self.head = nn.Sequential(*self.head)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = args.bn_mom

    def embed(self, x, s):
        x = self.features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        if self.concat_cell_type:
            x = torch.cat([x, s], dim=1)

        embedding = self.neck(x)
        return embedding

    def metric_classify(self, embedding):
        return self.arc_margin_product(embedding)

    def classify(self, embedding):
        return self.head(embedding)

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()

        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        encoded_labels = one_hot_encode(labels, logits.shape[-1])
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (encoded_labels * phi) + ((1.0 - encoded_labels) * cosine)
        output *= self.s
        loss = self.crit(output, encoded_labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ModelAndLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = Model(num_classes)
        self.metric_crit = ArcFaceLoss()
        self.crit = nn.CrossEntropyLoss()

    def train_forward(self, x, s, y, dummy_w=None):
        embedding = self.model.embed(x, s)

        metric_output = self.model.metric_classify(embedding)
        metric_loss = self.metric_crit(metric_output, y)

        output = self.model.classify(embedding)
        loss = None
        # Allow IRM trainin
        if dummy_w is not None:
            loss = self.crit(output * dummy_w, y)
        else:
            loss = self.crit(output, y)

        acc = accuracy_score(output.max(1)[1].cpu().numpy(), y.squeeze().cpu().numpy(), normalize=False)

        coeff = .2
        return loss * (1 - coeff) + metric_loss * coeff, acc

    def eval_forward(self, x, s):
        embedding = self.model.embed(x, s)
        output = self.model.classify(embedding)
        return output

    def embed(self, x, s):
        return self.model.embed(x, s)


class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.densenet161(pretrained=True, memory_efficient=True)
        # Make model work with six channels instead of 3
        self.model.features.conv0 = nn.Conv2d(6, 96, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # Make model predict the correct number of classes (1000 comes from output class count in ImageNet)
        self.out_layer = nn.Linear(1000, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_forward(self, x, s, y, dummy_w=None):

        # print(self.model)
        har =self.model(x)

        output = self.out_layer(har)


        loss = None
        if dummy_w is not None:
            loss = self.loss_fn(dummy_w * output, y)
        else:
            loss = self.loss_fn(output, y)

        num_correct = accuracy_score(output.max(1)[1].cpu().numpy(), y.squeeze().cpu().numpy(), normalize=False)
        return loss, num_correct

    def eval_forward(self, x, s):
        output = self.out_layer(self.model(x))
        return output


class MultitaskNet(nn.Module):
    '''This network learns a shared representation between cell types in the form of a Densenet,
    but learns a seperate classifier head for each cell type
    '''
    def __init__(self, num_classes, num_cell_types=3):
        super().__init__()

        self.model = torchvision.models.densenet161(pretrained=True, memory_efficient=True)
        # Make model work with six channels instead of 3
        self.model.features.conv0 = nn.Conv2d(6, 96, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # Make model predict the correct number of classes (1000 comes from output class count in ImageNet)
        self.out_layers = nn.ModuleList([nn.Linear(1000, num_classes) for i in range(num_cell_types)])
        self.loss_fn = nn.CrossEntropyLoss()

    def train_forward(self, x, s, y, dummy_w=None):
        representation = self.model(x)

        # Select classifier based on cell type
        # Note: this is assuming all items in the batch are the same cell type
        classifier_index = s[0, :].argmax()
        output = self.out_layers[classifier_index](representation)

        loss = None
        if dummy_w is not None:
            loss = self.loss_fn(dummy_w * output, y)
        else:
            loss = self.loss_fn(output, y)

        num_correct = accuracy_score(output.max(1)[1].cpu().numpy(), y.squeeze().cpu().numpy(), normalize=False)
        return loss, num_correct

    def eval_forward(self, x, s):
        with torch.no_grad():
            representation = self.model(x)

            prediction_sum = None
            for layer in self.out_layers:
                out = layer(representation)
                if prediction_sum is None:
                    prediction_sum = out
                else:
                    prediction_sum += out

        return prediction_sum
