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
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
