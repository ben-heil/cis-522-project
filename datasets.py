'''This file contains pytorch dataset objects and associated functions for processing image data'''

import numpy as np
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch

class RecursionDataset(Dataset):
    def __init__(self, csv_path, root_path):
        self.csv_df = pd.read_csv(csv_path)
        self.root_path = root_path
        self.len = self.csv_df.shape[0]
        self.channels = [1, 2, 3, 4, 5, 6]
        self.sites = [1, 2]

    def create_image_path(self, experiment, plate_num, well, site, channel):
        path = "%s/%s/Plate%s/%s_s%s_w%s.png" % (
            self.root_path, experiment, plate_num, well, site, channel)
        return path

    def load_image_from_path(self, path):
        img = np.asarray(Image.open(path))
        return img

    def create_img_tensor(self, experiment, plate_num, well):
        numpy_list = []

        generator_paths = (self.create_image_path(experiment,plate_num, well, site, channel) for channel in self.channels for site in self.sites)

        for img_path in generator_paths:
            try:
                numpy_list.append(self.load_image_from_path(img_path))
            except FileNotFoundError:
                print(img_path + "Does not exist")
                #some sites and channels do not exist.

        return torch.from_numpy(np.stack(numpy_list))

    def __getitem__(self, idx: int):
        row = self.csv_df.iloc[idx]
        id_code = row.loc['id_code']
        experiment = row.loc['experiment']
        plate_num = row.loc['plate']
        well = row.loc['well']
        sirna_label = row.loc['sirna']

        return_x = self.create_img_tensor(experiment, plate_num, well)

        return return_x, sirna_label

    def __len__(self):
        return self.len
