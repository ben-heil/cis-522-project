'''This file contains pytorch dataset objects and associated functions for processing image data'''

import numpy as np
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset


class RecursionDataset(Dataset):
    '''A dataset to process images from the broad image benchmark database'''

    def __init__(self, images: np.array, labels: np.array, encoder: preprocessing.LabelEncoder):
        '''Instantiate the dataset object

        Arguments
        ---------
        images:
            The array of images to make a dataset out of
        labels:
            The (encoded) array of image labels
        encoder:
            The encoder used to encode the image labels
        '''

        self.X = images
        self.y = labels
        self.encoder = encoder
        self.length = len(images)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.length
