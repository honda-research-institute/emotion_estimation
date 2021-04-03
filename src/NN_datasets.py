import torch
import numpy as np
from pathlib import Path
import deepdish as dd

import utils

# ECG dataset class for torch networks
class EcgDataset(torch.utils.data.Dataset):
    """ Class to create raw ECG datasets for training Self Supervised Learning models """

    def __init__(self, path, window_size, data_group=None):
        super(EcgDataset, self).__init__()

        self.path = path
        self.data_group = data_group
        self.window_size = window_size
        self.features, self.labels = self.import_data()

    def import_data(self):
        if self.data_group:
            data = dd.io.load(self.path, self.data_group)
        else:
            data = dd.io.load(self.path)

        if len(data['ECG'].shape) < 3:
            features = data['ECG'].reshape(-1, 1, self.window_size)
        else:
            features = data['ECG']
        labels = data['labels']

        return features, labels
         
    def __len__(self):
        return (self.features).shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ecg = self.features[idx, :]
        
        if len(self.labels) > 0:
            labels = self.labels[idx, :]
        else:
            labels = []

        sample = {'features': torch.tensor(ecg, dtype=torch.float, requires_grad=True), 'labels': torch.tensor(labels, dtype=torch.float, requires_grad=True)}
        return sample

# ECG feature dataset class for torch networks
class EcgFeatDataset(torch.utils.data.Dataset):
    """ Class to create raw ECG datasets for training Self Supervised Learning models """

    def __init__(self, path, data_group=None, balance_data=False):
        super(EcgFeatDataset, self).__init__()

        self.path = path
        self.data_group = data_group
        self.balance_data = balance_data
        self.features, self.labels = self.import_data()

    def import_data(self):
        if self.data_group:
            data = dd.io.load(self.path, self.data_group)
        else:
            data = dd.io.load(self.path)

        features = data['ECG']
        labels = data['labels']

        # There is a lot of data with neutral images in the training data
        if self.balance_data:
            bal_ind = utils.balance_labels(labels)
            features, labels = features[bal_ind, :], labels[bal_ind, :]

        return features, labels
         
    def __len__(self):
        return (self.features).shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ecg = self.features[idx, :]
        
        if len(self.labels) > 0:
            labels = self.labels[idx, :]
        else:
            labels = []

        sample = {'features': torch.tensor(ecg, dtype=torch.float, requires_grad=True), 'labels': torch.tensor(labels, dtype=torch.float, requires_grad=True)}
        return sample

# Multi feature dataset class for torch networks
class MultiFeatDataset(torch.utils.data.Dataset):
    """ Class to create raw multimodal dataset for training Self Supervised Learning models """

    def __init__(self, dataset, balance_data=False):
        super(MultiFeatDataset, self).__init__()

        # There is a lot of data with neutral images in the training data
        if balance_data:
            bal_ind = utils.balance_labels(dataset['labels'])
            self.features, self.labels = dataset['features'][bal_ind, :], dataset['labels'][bal_ind, :]
        else:
            self.features, self.labels = dataset['features'], dataset['labels']
         
    def __len__(self):
        return (self.features).shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if len(self.labels) > 0:
            labels = self.labels[idx, :]
        else:
            labels = []
            
        sample = {'features': torch.tensor(self.features[idx, :], dtype=torch.float, requires_grad=True), 
                  'labels': torch.tensor(labels, dtype=torch.float, requires_grad=True)}
        return sample