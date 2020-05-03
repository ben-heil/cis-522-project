'''This file contains pytorch dataset objects and associated functions for processing image data'''

import functools
import os
import sys

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import torch
from PIL import Image
from torch.utils.data import Dataset

@functools.lru_cache()
def load_metadata_df(csv_path):
    csv_df = pd.read_csv(csv_path, engine='python')

    return csv_df

class RecursionDataset(Dataset):
    def __init__(self, csv_path, root_path, sirna_encoder, mode='train', cell_type=None, sirnas_to_keep=None):
        self.sirna_encoder = sirna_encoder

        csv_path = csv_path.rstrip('/')
        if not os.path.exists(csv_path):
            print('Path {} does not exist'.format(csv_path))
            raise FileNotFoundError

        root_path = root_path.rstrip('/')

        self.csv_df = load_metadata_df(csv_path)

        # NORMALIZATION: Temporary location (feel free to move somewhere else)
        if args.normalization == 'plate':
          self.stats = pd.read_csv(os.path.join(args.data_dir, 'plate_stats.csv')) #ideally csv filename would not be hardcoded
        elif args.normalization == 'experiment':
          self.stats = pd.read_csv(os.path.join(args.data_dir, 'exp_stats.csv'))

        if sirnas_to_keep is not None:
            self.csv_df = self.csv_df[self.csv_df['sirna'].isin(sirnas_to_keep)]
            
        # Initialize cell type encoders
        self.cell_type_label_encoder = preprocessing.LabelEncoder()
        cell_type_encoded = self.cell_type_label_encoder.fit_transform(self.csv_df['cell_type'])
        self.cell_type_onehot_encoder = preprocessing.OneHotEncoder(categories='auto')
        self.cell_type_onehot_encoder.fit(cell_type_encoded.reshape(-1, 1))

        if cell_type is not None and cell_type.lower() != 'all':
            self.csv_df = self.csv_df[self.csv_df['cell_type'].str.match(cell_type)]

        self.csv_df = self.csv_df[self.csv_df['dataset'].str.match(mode)]
        self.csv_df = self.csv_df.reindex()

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

    def create_img_tensor(self, experiment, plate_num, well, site):
        numpy_list = []

        generator_paths = (self.create_image_path(experiment,plate_num, well, site, channel) for channel in self.channels)

        for img_path in generator_paths:
            try:
                numpy_list.append(self.load_image_from_path(img_path))
            except FileNotFoundError:
                print(img_path + " Does not exist")
                #some sites and channels do not exist.

        return torch.from_numpy(np.stack(numpy_list))

    def __getitem__(self, idx: int):
        row = self.csv_df.iloc[idx]
        experiment = row.loc['experiment']
        plate_num = row.loc['plate']
        well = row.loc['well']
        site = row.loc['site']
        sirna_label = [row.loc['sirna']]
        cell_type = [row.loc['cell_type']]

        cell_type = self.cell_type_label_encoder.transform(cell_type).reshape(-1, 1)
        cell_type = self.cell_type_onehot_encoder.transform(cell_type)
        cell_type = cell_type.toarray()

        sirna_label = self.sirna_encoder.transform(sirna_label).squeeze()

        return_x = self.create_img_tensor(experiment, plate_num, well, site)

        if args.normalization == 'plate':
          # Get all channel where mean & std values are located
          plate_bool = ( (self.stats['experiment']==experiment) & (self.stats['plate']==plate) )
          # Create stack of all channel values
          idx = self.stats.index[plate_bool] 

          pixel_mean = torch.tensor(self.stats['mean'][idx].to_numpy()) # Get mean
          pixel_std = torch.tensor(self.stats['std'][idx].to_numpy()) # Get std
          return_x = (return_x - pixel_mean.reshape(-1,1,1)) / pixel_std.reshape(-1,1,1) # Normalize
        elif args.normalization == 'experiment':
          # Get all channel where mean & std values are located
          exp_bool = (self.stats['experiment']==experiment)
          # Create stack of all channel values
          idx = self.stats.index[exp_bool]

          pixel_mean = torch.tensor(self.stats['mean'][idx].to_numpy()) # Get mean
          pixel_std = torch.tensor(self.stats['std'][idx].to_numpy()) # Get std
          return_x = (return_x - pixel_mean.reshape(-1,1,1)) / pixel_std.reshape(-1,1,1) # Normalize
        return return_x, cell_type, sirna_label

    def __len__(self):
        return self.len
