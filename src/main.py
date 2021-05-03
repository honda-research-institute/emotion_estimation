import sys
import os
import yaml
import time
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm 
import pickle
import pandas as pd 
import collections

import signal_transformation as sgtf
import utils
from utils import skip_run

import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from NN_models import SelfSupervisedNet, SelfSupervisedNet2, SelfSupervisedNetFeats, EcgNet, EmotionNet, EmotionNetLSTM
from NN_datasets import EcgDataset, EcgFeatDataset, MultiFeatDataset

from data_preprocessing import (read_raw_dreamer_dataset, read_raw_wesad_dataset, 
                                read_raw_hri_dataset, split_data_train_test_valid,
                                read_raw_hri_dataset_with_transition,
                                split_modalities_train_test_valid)
from feature_extraction import extract_gsr_features, extract_all_features
from Regression_models import train_test_regression_model, test_pretrained_regression_model

config = yaml.load(open(Path(__file__).resolve().parents[1] / 'config.yml'), Loader=yaml.SafeLoader)
window_len = config['freq'] * config['window_size']


##########################################################################
# ------ Self Supervised Network for learning ECG representations ------ #
##########################################################################
## transformation task params
noise_param = 15 #noise_amount
scale_param = 1.1 #scaling_factor
permu_param = 20 #permutation_pieces
tw_piece_param = 9 #time_warping_pieces
twsf_param = 1.05 #time_warping_stretch_factor

# SSL - self supervised learning
with skip_run('skip', 'prepare raw dataset for ECG-SSL') as check, check():
    # create data files
    read_raw_wesad_dataset(config['wesad']['load_path'], config['wesad']['interim'], config, save=True)
    read_raw_dreamer_dataset(config['dreamer']['load_path'], config['dreamer']['interim'], config, save=True)
    read_raw_hri_dataset(config['hri']['load_path'], config['hri']['interim_standardized'], config['hri']['percent_overlap'], config, save=True, standardize=True)
    
    wesad_data, dreamer_data, hri_data = [], [], []
    wesad_labels, dreamer_labels, hri_labels = [], [], []

    # load WESAD dataset
    wesad_dataset = dd.io.load(config['wesad']['interim'])
    for key in wesad_dataset.keys():
        wesad_data.append(wesad_dataset[key]['ECG'].reshape(-1, window_len))
    
    wesad_ecg = np.concatenate(wesad_data, axis=0)
    wesad_labels = np.arange(0, wesad_ecg.shape[0])

    # load DREAMER dataset
    dreamer_dataset = dd.io.load(config['dreamer']['interim'])
    for key in dreamer_dataset.keys():
        dreamer_data.append(dreamer_dataset[key]['ECG'].reshape(-1, window_len))
    
    dreamer_ecg = np.concatenate(dreamer_data, axis=0)
    dreamer_labels = np.arange(0, dreamer_ecg.shape[0])

    # load HRI dataset 
    hri_dataset = dd.io.load(config['hri']['interim_standardized'])   
    for subject in hri_dataset.keys():
        for event in ['event1', 'event2']:
            hri_data.append(hri_dataset[subject][event]['ECG'].reshape(-1, window_len))
            hri_labels.append(hri_dataset[subject][event]['labels']) 
    
    hri_data    = np.concatenate(hri_data, axis=0)
    hri_labels = np.concatenate(hri_labels, axis=0)

    # split the datasets into train, test and validation sets
    wesad_train, wesad_test, wesad_valid, _, _, _ = split_data_train_test_valid(wesad_ecg, wesad_labels, test_size=0.2, shuffle=True, random_state=1729)
    dreamer_train, dreamer_test, dreamer_valid, _, _, _ = split_data_train_test_valid(dreamer_ecg, dreamer_labels, test_size=0.2, shuffle=True, random_state=1729)
    hri_train, hri_test, hri_valid, _, _, _ = split_data_train_test_valid(hri_data, hri_labels, test_size=0.2, shuffle=True, random_state=1729)

    ecg_train = np.concatenate((wesad_train, dreamer_train, hri_train), axis=0)
    ecg_test  = np.concatenate((wesad_test, dreamer_test, hri_test), axis=0)
    ecg_valid = np.concatenate((wesad_valid, dreamer_valid, hri_valid), axis=0)

    Data = { "ECG_train": ecg_train,
             "ECG_test" : ecg_test,
             "ECG_valid": ecg_valid}

    dd.io.save(config['raw_data_pool'], Data)


# katsu: same as above but includes transition
with skip_run('skip', 'prepare raw dataset for ECG-SSL with transition') as check, check():
    # create data files
    read_raw_hri_dataset_with_transition(config['hri']['load_path'], config, save_path=config['hri']['interim_transition'], standardize=True, transition_delay_time=1.0)

    hri_data = []
    hri_labels = []

    # load HRI dataset
    hri_dataset = dd.io.load(config['hri']['interim_transition'])
    for subject in hri_dataset.keys():
        for event in ['event1', 'event2']:
            hri_data.append(hri_dataset[subject][event]['ECG'].reshape(-1, window_len))
            hri_labels.append(hri_dataset[subject][event]['labels'])

    hri_data    = np.concatenate(hri_data, axis=0)
    hri_labels = np.concatenate(hri_labels, axis=0)

    # split the datasets into train, test and validation sets
    ecg_train, ecg_test, ecg_valid, _, _, _ = split_data_train_test_valid(hri_data, hri_labels, test_size=0.5, shuffle=True, random_state=1729)

    Data = { "ECG_train": ecg_train,
             "ECG_test" : ecg_test,
             "ECG_valid": ecg_valid}

    print('saving pool data...')
    dd.io.save(config['raw_data_pool_transition'], Data)
    print('done')


# Transform each data point into 7 transformations and use it for SSL training
with skip_run('skip', 'prepare transform dataset') as check, check():
    Data = dd.io.load(config['raw_data_pool'])

    ecg_train, ecg_test, ecg_valid = [], [], []
    y_train, y_test, y_valid = [], [], []

    for i in range(Data['ECG_train'].shape[0]):
        signal = Data['ECG_train'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)
        
        ecg_train.append(tr_signal)
        y_train.append(tr_labels)
        print(i, '/', Data['ECG_train'].shape[0], ', tr_signal.shape:', tr_signal.shape, ', tr_labels.shape:', tr_labels.shape)
    
    ecg_train  = np.concatenate(ecg_train, axis=0)
    y_train    = np.concatenate(y_train, axis=0)

    for i in range(Data['ECG_test'].shape[0]):
        signal = Data['ECG_test'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)
        
        ecg_test.append(tr_signal)
        y_test.append(tr_labels)

    ecg_test  = np.concatenate(ecg_test, axis=0)
    y_test    = np.concatenate(y_test, axis=0)

    for i in range(Data['ECG_valid'].shape[0]):
        signal = Data['ECG_valid'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)
        
        ecg_valid.append(tr_signal)
        y_valid.append(tr_labels)

    ecg_valid  = np.concatenate(ecg_valid, axis=0)
    y_valid    = np.concatenate(y_valid, axis=0)

    train_data = { "ECG"    : ecg_train,
                   "labels" : y_train}
    test_data  = { "ECG"    : ecg_test,
                   "labels" : y_test}
    valid_data = { "ECG"    : ecg_valid,
                   "labels" : y_valid}

    dd.io.save(config['transformed_data'], {'train': train_data,
                                            'test' : test_data,
                                            'valid': valid_data})


# same as above but with transition
# save only half the frames because the original data size is too big
with skip_run('skip', 'prepare transform dataset with transition') as check, check():
    Data = dd.io.load(config['raw_data_pool_transition'])

    ecg_train, ecg_test, ecg_valid = [], [], []
    y_train, y_test, y_valid = [], [], []
    interval = 2

    print('transform training data...')
    for i in range(0, Data['ECG_train'].shape[0], interval):
        signal = Data['ECG_train'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)

        ecg_train.append(tr_signal)
        y_train.append(tr_labels)
        print(i, '/', Data['ECG_train'].shape[0])

    print('concatenate 1')
    ecg_train = np.concatenate(ecg_train, axis=0)
    print('concatenate 2')
    y_train = np.concatenate(y_train, axis=0)
    train_data = { "ECG"    : ecg_train,
                   "labels" : y_train}
    ecg_train = []
    y_train = []
    Data['ECG_train'] = []

    print('transform test data...')
    for i in range(0, Data['ECG_test'].shape[0], interval):
        signal = Data['ECG_test'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)

        ecg_test.append(tr_signal)
        y_test.append(tr_labels)
        print(i, '/', Data['ECG_test'].shape[0])

    ecg_test  = np.concatenate(ecg_test, axis=0)
    y_test    = np.concatenate(y_test, axis=0)
    test_data  = { "ECG"    : ecg_test,
                   "labels" : y_test}
    ecg_test = []
    y_test = []
    Data['ECG_test'] = []

    print('transform validation data...')
    for i in range(0, Data['ECG_valid'].shape[0], interval):
        signal = Data['ECG_valid'][i, :].reshape(-1,1)
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)

        ecg_valid.append(tr_signal)
        y_valid.append(tr_labels)

        print(i, '/', Data['ECG_valid'].shape[0])

    print('concatenate 1')
    ecg_valid  = np.concatenate(ecg_valid, axis=0)
    print('concatenate 2')
    y_valid    = np.concatenate(y_valid, axis=0)

    print('concatenate 3')
    valid_data = { "ECG"    : ecg_valid,
                   "labels" : y_valid}

    ecg_valid = []
    y_valid = []
    Data['ECG_valid'] = []

    print('save transformed data...')
    dd.io.save(config['transformed_data_transition'], {'train': train_data,
                                                       'test' : test_data,
                                                       'valid': valid_data})
    print('done')


# network module for self supervised representation learning using ECG
with skip_run('skip', 'train SSL model1') as check, check():
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/SSL_runs"):
        utils.makedirs("runs/SSL_runs")

    if ~os.path.exists("models/SSL_models"):
        utils.makedirs("models/SSL_models")

    batch_size = 64
    window_size = config['freq'] * config['window_size']
    task_weights = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]

    train_data = EcgDataset(config['transformed_data'], window_size, data_group='/train')
    test_data  = EcgDataset(config['transformed_data'], window_size, data_group='/test')
    valid_data = EcgDataset(config['transformed_data'], window_size, data_group='/valid')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    net = SelfSupervisedNet(device, config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    net.train_model(train_dataloader, valid_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=200, task_weights=task_weights)

# modified network module for self supervised representation learning using ECG
with skip_run('skip', 'train SSL model2') as check, check():
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/SSL_runs"):
        utils.makedirs("runs/SSL_runs")

    if ~os.path.exists("models/SSL_models"):
        utils.makedirs("models/SSL_models")

    batch_size = 64
    window_size = config['freq'] * config['window_size']
    task_weights = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]

    train_data = EcgDataset(config['transformed_data'], window_size, data_group='/train')
    test_data  = EcgDataset(config['transformed_data'], window_size, data_group='/test')
    valid_data = EcgDataset(config['transformed_data'], window_size, data_group='/valid')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    net = SelfSupervisedNet2(device, config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    net.train_model(train_dataloader, valid_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=200, task_weights=task_weights)


# same as above but with transition
with skip_run('run', 'train SSL model2 with transition') as check, check():
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/SSL_runs"):
        utils.makedirs("runs/SSL_runs")

    if ~os.path.exists("models/SSL_models"):
        utils.makedirs("models/SSL_models")

    batch_size = 64
    window_size = config['freq'] * config['window_size']
    task_weights = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]

    train_data = EcgDataset(config['transformed_data_transition'], window_size, data_group='/train')
    test_data  = EcgDataset(config['transformed_data_transition'], window_size, data_group='/test')
    valid_data = EcgDataset(config['transformed_data_transition'], window_size, data_group='/valid')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    net = SelfSupervisedNet2(device, config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    net.train_model(train_dataloader, valid_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=200, task_weights=task_weights)


# Test the desired self supervised learning model 
with skip_run('skip', 'test SSL models') as check, check():
    device = torch.device("cpu")
    
    # Give the input to test either of the models
    test_model = '1'

    if test_model == '1':
        model = SelfSupervisedNet(device, config)
        file = 'model1_0/net_80.pth'
    elif test_model == '2':
        model = SelfSupervisedNet2(device, config)
        file = 'model2_1/net_200.pth'
    else:
        print('There are only two types of SSL models')
        sys.exit() 
        
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    window_size = config['freq'] * config['window_size']
    test_data = EcgDataset(config['transformed_data'], window_size, data_group='/test')

    # train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    # for file in os.listdir(config['torch']['SSL_models']):

    # if not os.path.isdir(file):
    checkpoint = torch.load(config['torch']['SSL_models'] + file, map_location=device)
    results = model.validate_model(test_dataloader, optimizer=optimizer, checkpoint=checkpoint, device=device)

    print("Epoch: {}, Accuracy of model {}: {}".format(results['epoch'], test_model, results['accuracy']))


##########################################################################
# ---------- Dataset for downstream Emotion recognition task ----------- #
##########################################################################
# Prepare the datasets for training the downstream EMOTION recognition task
with skip_run('skip', 'prepare HRI training and testing datasets') as check, check():
    # parse raw HRI data
    read_raw_hri_dataset(config['hri']['load_path'], config['hri']['interim_raw'], config['hri']['percent_overlap'], config, save=True, standardize=False)

    # parse raw HRI data and standardize it per individual
    read_raw_hri_dataset(config['hri']['load_path'], config['hri']['interim_standardized'], config['hri']['percent_overlap'], config, save=True, standardize=True)
   
with skip_run('skip', 'pool HRI standardized data') as check, check():
    data = dd.io.load(config['hri']['interim_standardized'])
    hri_ecg, hri_emg, hri_gsr, hri_ppg, hri_rsp, hri_labels = [], [], [], [], [], []

    for subject in data.keys():
        print(subject)
        for event in data[subject]:
            hri_ecg.append(data[subject][event]['ECG'])
            hri_emg.append(data[subject][event]['EMG'])
            hri_gsr.append(data[subject][event]['GSR'])
            hri_ppg.append(data[subject][event]['PPG'])
            hri_rsp.append(data[subject][event]['RSP'])

            hri_labels.append(data[subject][event]['labels']) 
    
    hri_ecg    = np.concatenate(hri_ecg, axis=0)
    hri_emg    = np.concatenate(hri_emg, axis=0)
    hri_gsr    = np.concatenate(hri_gsr, axis=0)
    hri_ppg    = np.concatenate(hri_ppg, axis=0)
    hri_rsp    = np.concatenate(hri_rsp, axis=0)
    hri_labels = np.concatenate(hri_labels, axis=0)

    Data_dic   = {'ECG': hri_ecg,
                  'EMG': hri_emg,
                  'GSR': hri_gsr,
                  'PPG': hri_ppg,
                  'RSP': hri_rsp}

    train_data, test_data, valid_data = split_modalities_train_test_valid(Data_dic, hri_labels, test_size=0.20, shuffle=True, random_state=1729)

    dd.io.save(config['hri_pooled'], {'train': train_data,
                                      'test' : test_data,
                                      'valid': valid_data})

# First 3 and Last 3 samples for each event dataset  
with skip_run('skip', 'prepare first and last sample datasets') as check, check():
    # first sample dataset 
    read_raw_hri_dataset(config['hri']['load_path'], config['hri']['interim_first_samp'], config['hri']['percent_overlap'], config, save=True, standardize=True, pick_sample='first')

    # last sample dataset
    read_raw_hri_dataset(config['hri']['load_path'], config['hri']['interim_last_samp'], config['hri']['percent_overlap'], config, save=True, standardize=True, pick_sample='last')

 
##########################################################################
#---------------- Physiological data feature extraction -----------------#
##########################################################################
with skip_run('skip', 'extract ECG-SSL features for each individual') as check, check():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch size here is provided while extracting the features to prevent CUDA out of memory issue
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    ckpt_file  = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    data_dic = dd.io.load(config['hri']['interim_standardized'])
    
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            data_group = str('/data/' + subject + '/' + event)
            train_data = EcgDataset(config['hri']['interim_standardized'], window_size, data_group=data_group)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            feat_train, labels_train = net.return_SSL_feats(train_dataloader)

            data[event] = {"ECG": feat_train, "labels": labels_train}

        Data[subject] = data
    dd.io.save(config['hri_ECG_SSL_feats'], Data)


# katsu: same as above but with transition
with skip_run('skip', 'extract ECG-SSL features for each individual with transition') as check, check():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # batch size here is provided while extracting the features to prevent CUDA out of memory issue
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    ckpt_file  = 'model2_1/net_200.pth'
    print('loading model')
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    print('loading data')
    data_dic = dd.io.load(config['hri']['interim_transition'])

    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            print('subject:', subject, ', event:', event)
            data_group = str('/data/' + subject + '/' + event)
            train_data = EcgDataset(config['hri']['interim_transition'], window_size, data_group=data_group)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            feat_train, labels_train = net.return_SSL_feats(train_dataloader)

            data[event] = {"ECG": feat_train, "labels": labels_train}

        Data[subject] = data
    dd.io.save(config['hri_ECG_SSL_feats_transition'], Data)


# Extract the handcrafted features for each individual
with skip_run('skip', 'Extract all handcrafted features from each modality for individual') as check, check():
    data_dic = dd.io.load(config['hri']['interim_standardized'])
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            features = extract_all_features(data_dic[subject][event], config)
            data[event] = features

        Data[subject] = data 
    dd.io.save(config['hri_feats'], Data)


# katsu: same as above but with transition
# currently unsable because this generates a bunch of errors
with skip_run('skip', 'Extract all handcrafted features from each modality for individual with transition') as check, check():
    data_dic = dd.io.load(config['hri']['interim_transition'])
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            features = extract_all_features(data_dic[subject][event], config)
            data[event] = features

        Data[subject] = data

    dd.io.save(config['hri_feats_transition'], Data)


# Extract the ECG features from the Self Supervised Net (N samples x 128 features)
with skip_run('skip', 'Extract ECG-SSL feature dataset from pooled ECG data') as check, check():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch size here is provided while extracting the features to prevent CUDA out of memory issue
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    train_data = EcgDataset(config['hri_pooled'], window_size, data_group='/train')
    test_data  = EcgDataset(config['hri_pooled'], window_size, data_group='/test')
    valid_data = EcgDataset(config['hri_pooled'], window_size, data_group='/valid')
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    ckpt_file  = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    feat_train, labels_train = net.return_SSL_feats(train_dataloader)
    feat_test, labels_test   = net.return_SSL_feats(test_dataloader)
    feat_valid, labels_valid = net.return_SSL_feats(valid_dataloader)

    dd.io.save(config['hri_ECG_SSL_feats_pooled'], {'train': {"ECG" : feat_train, "labels" : labels_train},
                                             'test' : {"ECG" : feat_test,  "labels"  : labels_test},
                                             'valid': {"ECG" : feat_valid, "labels" : labels_valid}})
# Extract the handcrafted features for each modality
with skip_run('skip', 'Extract all handcrafted features from each modality of pooled data') as check, check():
    window_size = config['freq'] * config['window_size']
    train_data = dd.io.load(config['hri_pooled'], '/train')
    test_data  = dd.io.load(config['hri_pooled'], '/test')
    valid_data = dd.io.load(config['hri_pooled'], '/valid')

    valid_feat = extract_all_features(valid_data, config)
    train_feat = extract_all_features(train_data, config)
    test_feat  = extract_all_features(test_data, config)

    dd.io.save(config['hri_feats_pooled'], {'train': train_feat,
                                     'valid': valid_feat,
                                     'test' : test_feat})


# First and Last sample dataset features   
with skip_run('skip', 'Extract ECG-SSL features for each individual for first and last sample dataset') as check, check():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch size here is provided while extracting the features to prevent CUDA out of memory issue
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    ckpt_file  = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    # first sample database
    data_dic = dd.io.load(config['hri']['interim_first_samp'])
    
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            data_group = str('/data/' + subject + '/' + event)
            train_data = EcgDataset(config['hri']['interim_first_samp'], window_size, data_group=data_group)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            feat_train, labels_train = net.return_SSL_feats(train_dataloader)

            data[event] = {"ECG": feat_train, "labels": labels_train}

        Data[subject] = data
    dd.io.save(config['hri_ECG_SSL_feats_first_samp'], Data)

    # last sample database
    data_dic = dd.io.load(config['hri']['interim_last_samp'])
    
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            data_group = str('/data/' + subject + '/' + event)
            train_data = EcgDataset(config['hri']['interim_last_samp'], window_size, data_group=data_group)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
            feat_train, labels_train = net.return_SSL_feats(train_dataloader)

            data[event] = {"ECG": feat_train, "labels": labels_train}

        Data[subject] = data
    dd.io.save(config['hri_ECG_SSL_feats_last_samp'], Data)
with skip_run('skip', 'Extract all handcrafted features from each modality for individual for first and last sample dataset') as check, check():
    data_dic = dd.io.load(config['hri']['interim_first_samp'])
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            features = extract_all_features(data_dic[subject][event], config)
            data[event] = features

        Data[subject] = data 
    dd.io.save(config['hri_feats_first_samp'], Data)

    # last sample features
    data_dic = dd.io.load(config['hri']['interim_last_samp'])
    Data = collections.defaultdict(dict)
    for subject in data_dic:
        data = collections.defaultdict(dict)
        for event in data_dic[subject]:
            features = extract_all_features(data_dic[subject][event], config)
            data[event] = features

        Data[subject] = data 
    dd.io.save(config['hri_feats_last_samp'], Data)
       
       
##########################################################################
#----- Train conventional Random Forest regression individual features -----#
##########################################################################
with skip_run('skip', 'train RF Regressor on the ECG-SSL features') as check, check():
    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/')

    train_data = dd.io.load(config['hri_ECG_SSL_feats_pooled'], '/train')
    test_data  = dd.io.load(config['hri_ECG_SSL_feats_pooled'], '/test')
    valid_data = dd.io.load(config['hri_ECG_SSL_feats_pooled'], '/valid')

    modality = 'ECG-SSL'
    # Linear regression and SVR are not performing well in estimating the valence-arousal levels
    # regressor = MultiOutputRegressor(Ridge()) 
    # regressor = MultiOutputRegressor(SVR())

    # Until now RandomForest has done best job in recognizing the valence-arousal levels
    regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))

    train_dataset = {'features': train_data['ECG'],
                     'labels': train_data['labels']}
                     
    test_dataset  = {'features': np.concatenate((test_data['ECG'], valid_data['ECG']), axis=0),
                     'labels': np.concatenate((test_data['labels'], valid_data['labels']), axis=0)}
    
    model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/' + modality + '.pkl'
    pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/'+ modality + '.png'
    results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/' + modality + '.csv'

    train_test_regression_model(modality, train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)

with skip_run('skip', 'train RF Regressor using hand-crafted features for each modalities') as check, check():
    
    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/')

    train_data = dd.io.load(config['hri_feats_pooled'], '/train')
    test_data  = dd.io.load(config['hri_feats_pooled'], '/test')
    valid_data = dd.io.load(config['hri_feats_pooled'], '/valid')

    # Linear regression and SVR are not performing well in estimating the valence-arousal levels
    # regressor = MultiOutputRegressor(Ridge()) 
    # regressor = MultiOutputRegressor(SVR())

    for modality in config['hri']['ch_names']:
        
        if modality in train_data.keys():
            # Until now RandomForest has done best job in recognizing the valence-arousal levels
            regressor = (RandomForestRegressor(n_estimators=100, max_depth=30))
            
            model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/' + modality + '.pkl'
            pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/' + modality + '.png'
            results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/' + modality + '.csv'

            # Creating dataset after taking care of additional dimension at axis=1
            train_dataset = {'features': train_data[modality].reshape(train_data[modality].shape[0], -1),
                            'labels': train_data['labels']} 
            test_dataset  = {'features': np.concatenate((test_data[modality].reshape(test_data[modality].shape[0], -1),
                                                        valid_data[modality].reshape(valid_data[modality].shape[0], -1)), axis=0),
                            'labels': np.concatenate((test_data['labels'], valid_data['labels']), axis=0)}

            train_test_regression_model(modality, train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)


##########################################################################
# ----- Downstream Emotion recognition on pooled ECG-SSL features ----- #
##########################################################################
with skip_run('skip', 'train the EcgNet with ECG-SSL1') as check, check():
    use_model = '1'
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/EMOTION_runs"):
        utils.makedirs("runs/EMOTION_runs")

    if ~os.path.exists("models/EMOTION_models"):
        utils.makedirs("models/EMOTION_models")

    batch_size = 128
    window_size = config['freq'] * config['window_size']

    train_data = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/train', balance_data=True)
    test_data  = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/test', balance_data=False)
    valid_data = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/valid', balance_data=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    torch_file = 'model1_0/net_80.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + torch_file, map_location=device)

    net = EcgNet(device=device, load_model=True, checkpoint=checkpoint, config=config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    net.train_model(train_dataloader, valid_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, load_model=True, use_model=use_model, epochs=500)

with skip_run('skip', 'train the EcgNet with ECG-SSL2') as check, check():
    use_model = '2'
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/EMOTION_runs"):
        utils.makedirs("runs/EMOTION_runs")

    if ~os.path.exists("models/EMOTION_models"):
        utils.makedirs("models/EMOTION_models")

    batch_size = 128
    window_size = config['freq'] * config['window_size']

    train_data = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/train', balance_data=True)
    test_data  = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/test', balance_data=False)
    valid_data = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/valid', balance_data=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    torch_file = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + torch_file, map_location=device)

    net = EcgNet(device=device, load_model=True, checkpoint=checkpoint, config=config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    net.train_model(train_dataloader, valid_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, load_model=True, use_model=use_model, epochs=500)

with skip_run('skip', 'test EcgNet') as check, check():
    device = torch.device("cpu")    
    window_size = config['freq'] * config['window_size']

    train_data = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/train', balance_data=True)
    test_data  = EcgFeatDataset(config['hri_ECG_SSL_feats_pooled'], data_group='/test', balance_data=False)

    file = 'model2_5/net_500.pth'
    model = EcgNet(device=device, load_model=False, checkpoint=[], config=config)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    checkpoint = torch.load(config['torch']['EMOTION_models'] + file, map_location=device)
    _, ax = plt.subplots(1, 2)

    train_results = model.validate_model(train_dataloader, optimizer=optimizer, checkpoint=checkpoint, device=device, ax=ax[0])
    test_results  = model.validate_model(test_dataloader, optimizer=optimizer, checkpoint=checkpoint, device=device, ax=ax[1])

    print("Train Epoch: {0:d}, Arousal : {1:.3f}, Valence : {2:.3f}, MSE : {3:.3f}, R2score : {4:.3f} of the model".format(train_results['epoch'], train_results['Arousal'], train_results['Valence'], train_results['mse'], train_results['r2score']))
    print("Test  Epoch: {0:d}, Arousal : {1:.3f}, Valence : {2:.3f}, MSE : {3:.3f}, R2score : {4:.3f} of the model".format(test_results['epoch'], test_results['Arousal'], test_results['Valence'], test_results['mse'], test_results['r2score']))


##########################################################################
#---- Downstream Multi-modal emotion recognition using pooled data ----#
##########################################################################
with skip_run('skip', 'Run the multimodal regression using RF and DNN') as check, check():
    
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/EMOTION_runs"):
        utils.makedirs("runs/EMOTION_runs")

    if ~os.path.exists("models/EMOTION_models"):
        utils.makedirs("models/EMOTION_models")

    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats'])
    data = dd.io.load(config['hri_feats'])

    Features, Labels = [], []
    for sub in data:
        for event in data[sub]:
            features, labels = [], []
            labels = data[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))
            
            features = np.concatenate(features, axis=1)

            Features.append(features)
            Labels.append(labels)

    Features = np.concatenate(Features, axis=0)
    Labels   = np.concatenate(Labels, axis=0)

    # if balance_train_data:
    #     bal_ind = utils.balance_labels(train_data['labels'])
    # else:
    #     bal_ind = np.arange(train_data['labels'].shape[0])
    #  

    train, test, _, y_train, y_test, _ = split_data_train_test_valid(Features, Labels, test_size=0.2, shuffle=True, random_state=None)
    
    train_dataset = {'features': train,
                     'labels': y_train}
    
    test_dataset = {'features': test,
                     'labels': y_test}
    
    # Random forest regression
    print('Now training Random Forest')
    regressor = (RandomForestRegressor(n_estimators=200, max_depth=30))
    model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/multimodal_regression/Multi_pooled_RF.pkl'
    pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/multimodal_regression/Multi_pooled_RF.png'
    results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/multimodal_regression/Multi_pooled_RF.csv'

    train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)

    # Train DNN
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    train_data = MultiFeatDataset(train_dataset, balance_data=False)
    test_data  = MultiFeatDataset(test_dataset, balance_data=False)
    # valid_data = MultiFeatDataset(valid_dataset, balance_data=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))


    net = EmotionNet(num_feats=train_dataset['features'].shape[1], device=device, config=config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = None
    net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=500)

with skip_run('skip', 'Run multimodal regression using RF and DNN and test it on 2 new subjects') as check, check():
    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/EMOTION_runs"):
        utils.makedirs("runs/EMOTION_runs")

    if ~os.path.exists("models/EMOTION_models"):
        utils.makedirs("models/EMOTION_models")

    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats'])
    data = dd.io.load(config['hri_feats'])

    train_features, train_labels = [], []
    for sub in ['S1', 'S2', 'S4', 'S5', 'S6', 'S8', 'S9', 'S10', 'S11']:
        for event in data[sub]:
            features, labels = [], []
            labels = data[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))
            
            features = np.concatenate(features, axis=1)

            train_features.append(features)
            train_labels.append(labels)

    train_features = np.concatenate(train_features, axis=0)
    train_labels   = np.concatenate(train_labels, axis=0)

    train_dataset = {'features': train_features,
                     'labels': train_labels}

    test_features, test_labels = [], []
    for sub in ['S3', 'S7']:
        for event in data[sub]:
            features, labels = [], []
            labels = data[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))
            
            features = np.concatenate(features, axis=1)

            test_features.append(features)
            test_labels.append(labels)

    test_features = np.concatenate(test_features, axis=0)
    test_labels   = np.concatenate(test_labels, axis=0)    

    test_dataset = {'features': test_features,
                     'labels': test_labels}
    
    # Random forest regression
    print('--------Now Training Random For Inter-subject------')
    regressor = (RandomForestRegressor(n_estimators=200, max_depth=30))

    model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/multimodal_regression/Multi_intersubject_RF.pkl'
    pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/multimodal_regression/Multi_intersubject_RF.png'
    results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/multimodal_regression/Multi_intersubject_RF.csv'

    train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)

    # Train DNN
    batch_size = 128
    window_size = config['freq'] * config['window_size']

    train_data = MultiFeatDataset(train_dataset, balance_data=False)
    test_data  = MultiFeatDataset(test_dataset, balance_data=False)
    # valid_data = MultiFeatDataset(valid_dataset, balance_data=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Train the model on: {}'.format(device))

    torch_file = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + torch_file, map_location=device)

    net = EmotionNet(num_feats=train_dataset['features'].shape[1], device=device, config=config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = None
    net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=500)

# 4) Test how small the training dataset could be after which the dataset diverges
# test No. 22 in Katsu's note
with skip_run('skip', 'Multimodal regression - split features into train and test sets based on increasing valence levels') as check, check():
    # train RF and DNN model if True, else, test the pre-trained models
    train_model = True
    
    data = dd.io.load(config['hri_feats'])
    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats'])

    Features, Labels = [], []

    target_subs = ['S5']
    #modalities = ['EMG', 'GSR', 'PPG', 'RSP']
    modalities = []
    #for sub in data:
    for sub in target_subs:
        print('sub:', sub)
        for event in data[sub]:
            features, labels = [], []
            labels = data[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in modalities:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))
            
            features = np.concatenate(features, axis=1)

            Features.append(features)
            Labels.append(labels)

    Features = np.concatenate(Features, axis=0)
    Labels   = np.concatenate(Labels, axis=0)

    print('Features shape (all):', Features.shape)
    print('Labels shape (all):', Labels.shape)

    # get the sort indices according to increasing values of valence 
    val_sort_ind = np.argsort(Labels[:, 1])

    Features = Features[val_sort_ind, :]
    Labels   = Labels[val_sort_ind, :]

    ind      = np.arange(0, Labels.shape[0])

    #change_ind = np.concatenate([[0], change_ind, [ind[-1]]])
    # every 11 frames
    change_ind = np.concatenate([np.array(range(0, Labels.shape[0], 11)), [Labels.shape[0]]])
    num_cat    = change_ind.shape[0] - 1
    print('Number of labels: {}'.format(Labels.shape[0]))
    print('Number of image categories: {}'.format(num_cat))

    #step_sizes = [2, 3, 4, 5, 10, 20, 50]
    #NN_model_dic = {'2': 0, '3': 1, '4':2, '5':3, '10':4, '20':5, '50':6}
    step_sizes = [2, 3, 4, 5, 10, 20]
    NN_model_dic = {'2': 0, '3': 1, '4':2, '5':3, '10':4, '20':5}

    per_train_data, RF_ars_mean, RF_val_mean, NN_ars_mean, NN_val_mean = [], [], [], [], []
    RF_ars_err, RF_val_err, NN_ars_err, NN_val_err = [], [], [], []

    for step_size in step_sizes:        
        categories = np.arange(num_cat) + 1
        train_categories = np.arange(1, num_cat, step_size)
        test_categories = list(set(categories) ^ set(train_categories))

        train_ind = []
        for i in train_categories:
            train_ind.append(range(change_ind[i-1], change_ind[i]))
        train_ind = np.concatenate(train_ind, axis=0)

        test_ind = []
        for i in test_categories:
            test_ind.append(range(change_ind[i-1], change_ind[i]))
        test_ind = np.concatenate(test_ind, axis=0)
        
        train_dataset = {'features': Features[train_ind, :],
                        'labels': Labels[train_ind, :]}
        
        test_dataset = {'features': Features[test_ind, :],
                        'labels': Labels[test_ind, :]}
        
        # Random forest regression
        # print('Now training Random Forest with step size: {}'.format(step_size))

        # model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/multimodal_regression/Multi_category_RF_' + str(step_size) + '.pkl'
        # pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/multimodal_regression/Multi_category_RF_' + str(step_size) + '.png'
        # results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/multimodal_regression/Multi_category_RF_' + str(step_size) + '.csv'

        # if train_model:
        #     regressor = (RandomForestRegressor(n_estimators=200, max_depth=30))
        #     RF_trn_ars, RF_trn_val, RF_tst_ars, RF_tst_val = train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)
        # else:
        #     regressor = pickle.load(open(model_save_path, 'rb'))
        #     RF_trn_ars, RF_trn_val, RF_tst_ars, RF_tst_val = test_pretrained_regression_model(modality, train_dataset, test_dataset, regressor, clean_features=True)

        # Train DNN
        batch_size = 128
        window_size = config['freq'] * config['window_size']

        train_data = MultiFeatDataset(train_dataset, balance_data=False)
        test_data  = MultiFeatDataset(test_dataset, balance_data=False)

        # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        # test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        # for LSTM
        batch_size = 64
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=11*batch_size, shuffle=False)
        test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=11*batch_size, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #net = EmotionNet(num_feats=train_dataset['features'].shape[1], device=device, config=config)
        net = EmotionNetLSTM(num_feats=train_dataset['features'].shape[1], seq_len=11, device=device, config=config)
        net.to(device)

        # see if an exponential decay learning rate scheduler is required
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = None
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        if train_model:
            print('Train the model on: {}'.format(device))
            NN_trn_ars, NN_trn_val, NN_tst_ars, NN_tst_val = net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=1000)
        else:
            torch_file = 'MultiModal_model_' + str(NN_model_dic[str(step_size)]) + '/net_500.pth'

            checkpoint = torch.load(config['torch']['EMOTION_models'] + torch_file, map_location=device)
            fig, ax = plt.subplots(1, 2)
            fig.suptitle('DNN')

            trn_results = net.validate_model(train_dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=ax[0])
            tst_results = net.validate_model(test_dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=ax[1])

            NN_trn_ars = trn_results['Arousal']
            NN_trn_val = trn_results['Valence'] 
            NN_tst_ars = tst_results['Arousal'] 
            NN_tst_val = tst_results['Valence']

        print(RF_trn_ars.shape, RF_trn_val.shape, RF_tst_ars.shape, RF_tst_val.shape)
        print(NN_trn_ars.shape, NN_trn_val.shape, NN_tst_ars.shape, NN_tst_val.shape)

        n_bins = 10
        fig1, ax1 = plt.subplots(2, 2, sharex=True, sharey=True)
        ax1[0, 0].hist(RF_tst_ars, bins=n_bins)
        ax1[0, 0].set_title('Arousal-RF ({})'.format(step_size))
        ax1[0, 0].set_xlim([0, 4])
        ax1[0, 0].set_ylim([0, 6000])

        ax1[0, 1].hist(NN_tst_ars, bins=n_bins)
        ax1[0, 1].set_title('Arousal-DNN ({})'.format(step_size))
        ax1[0, 1].set_xlim([0, 4])
        ax1[0, 1].set_ylim([0, 6000])

        ax1[1, 0].hist(RF_tst_val, bins=n_bins)
        ax1[1, 0].set_title('Valence-RF ({})'.format(step_size))
        ax1[1, 0].set_xlim([0, 4])
        ax1[1, 0].set_ylim([0, 6000])

        ax1[1, 1].hist(NN_tst_val, bins=n_bins)
        ax1[1, 1].set_title('Valence-DNN ({})'.format(step_size))
        ax1[1, 1].set_xlim([0, 4])
        ax1[1, 1].set_ylim([0, 6000])
        fig1.suptitle('Error histogram')

        # fig2, ax2 = plt.subplots(2, 2, sharex=True, sharey=True)
        # ax2[0, 0].hist(RF_trn_val, bins=n_bins)
        # ax2[0, 0].set_title('RF-Train')

        # ax2[0, 1].hist(NN_trn_val, bins=n_bins)
        # ax2[0, 1].set_title('DNN-Train')

        # ax2[1, 0].hist(RF_tst_val, bins=n_bins)
        # ax2[1, 0].set_title('RF-Test')

        # ax2[1, 1].hist(NN_tst_val, bins=n_bins)
        # ax2[1, 1].set_title('DNN-Test')
        # fig2.suptitle('Valence absolute error')

        per_train_data.append((train_categories.shape[0] * 100 / num_cat))
        RF_ars_mean.append(np.mean(RF_tst_ars))
        RF_val_mean.append(np.mean(RF_tst_val))
        NN_ars_mean.append(np.mean(NN_tst_ars))
        NN_val_mean.append(np.mean(NN_tst_val)) 

        # RF_ars_err.append(np.array([np.mean(RF_tst_ars) - np.min(RF_tst_ars), np.max(RF_tst_ars) - np.mean(RF_tst_ars)]).reshape(2, 1))
        # RF_val_err.append(np.array([np.mean(RF_tst_val) - np.min(RF_tst_val), np.max(RF_tst_val) - np.mean(RF_tst_val)]).reshape(2, 1))
        # NN_ars_err.append(np.array([np.mean(NN_tst_ars) - np.min(NN_tst_ars), np.max(NN_tst_ars) - np.mean(NN_tst_ars)]).reshape(2, 1))
        # NN_val_err.append(np.array([np.mean(NN_tst_val) - np.min(NN_tst_val), np.max(NN_tst_val) - np.mean(NN_tst_val)]).reshape(2, 1)) 

        RF_ars_err.append(np.std(RF_tst_ars))
        RF_val_err.append(np.std(RF_tst_val))
        NN_ars_err.append(np.std(NN_tst_ars))
        NN_val_err.append(np.std(NN_tst_val))

    # RF_ars_err = np.concatenate(RF_ars_err, axis=1)
    # RF_val_err = np.concatenate(RF_val_err, axis=1)
    # NN_ars_err = np.concatenate(NN_ars_err, axis=1)
    # NN_val_err = np.concatenate(NN_val_err, axis=1)
    per_train_data = np.array(per_train_data)

    fig, ax = plt.subplots(1, 2)
    ax[0].errorbar(per_train_data, RF_ars_mean, yerr=RF_ars_err, ecolor='g', mec='g', mfc='g', ms=10, capsize=4, fmt='s', label='RF')
    ax[0].set_title('Arousal test error (RF)')
    ax[0].set_ylabel('Mean Absolute Error')
    ax[0].set_xlabel('Percentage training data')
    ax[0].set_ylim([0, 2.5])
    ax[0].set_xlim([0, 60])
    # ax[0].legend()

    ax[1].errorbar(per_train_data, RF_val_mean, yerr=RF_val_err, ecolor='g', mec='g', mfc='g', ms=10, capsize=4, fmt='s', label='RF')
    ax[1].set_title('Valence test error (RF)')
    # ax[1].set_ylabel('Mean Absolute Error')
    ax[1].set_xlabel('Percentage training data')
    ax[1].set_ylim([0, 2.5])
    ax[1].set_xlim([0, 60])

    fig, ax = plt.subplots(1, 2)
    ax[0].errorbar(per_train_data, NN_ars_mean, yerr=NN_ars_err, ecolor='k', mec='k', mfc='k', ms=10, capsize=4, fmt='s', label='DNN')
    ax[0].set_title('Arousal test error (DNN)')
    ax[0].set_ylabel('Mean Absolute Error')
    ax[0].set_xlabel('Percentage training data')
    ax[0].set_ylim([0, 2.5])
    ax[0].set_xlim([0, 60])

    ax[1].errorbar(per_train_data, NN_val_mean, yerr=NN_val_err, ecolor='k', mec='k', mfc='k', ms=10, capsize=4, fmt='s', label='DNN')
    ax[1].set_title('Valence test error (DNN)')
    # ax[1].set_ylabel('Mean Absolute Error')
    ax[1].set_xlabel('Percentage training data')
    ax[1].set_ylim([0, 2.5])
    ax[1].set_xlim([0, 60])

    print("Percentages of training images used: {}".format(per_train_data))
    
##########################################################################
# ------ Error plots  ------#
##########################################################################
with skip_run('skip', 'Plot MAE and STD of trained regressors on the test data') as check, check():

    modalities = ['ECG-SSL', 'ECG', 'EMG', 'GSR', 'PPG']

    for dataset in ['Train', 'Test']:
        if dataset == 'Train':
            ecolor='b'
            x = [1, 3, 5, 7, 9]
        else:
            ecolor='r'
            x = [1.5, 3.5, 5.5, 7.5, 9.5] 

        arousal_mean, valence_mean, total_mean = [], [], []
        arousal_std, valence_std, total_std = [], [], []
        for modality in modalities:
            results_load_path = str(Path(__file__).parents[1]) + '/results/Regression/' + modality + '.csv'
            df = pd.read_csv(results_load_path,
                        delimiter=',',
                        usecols=['Dataset', 'Affect', 'MAE', 'STD'])

            arousal_mean.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Arousal'), 'MAE'].to_numpy())
            arousal_std.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Arousal'), 'STD'].to_numpy())

            valence_mean.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Arousal'), 'MAE'].to_numpy())
            valence_std.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Arousal'), 'STD'].to_numpy())

            if dataset == 'Test':
                total_mean.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Total'), 'MAE'].to_numpy())
                total_std.append(df.loc[(df['Dataset']==dataset) & (df['Affect']=='Total'), 'STD'].to_numpy())
        
        arousal_mean = np.array(arousal_mean).reshape(-1, )
        arousal_std  = np.array(arousal_std).reshape(-1, )
        valence_mean = np.array(valence_mean).reshape(-1, )
        valence_std  = np.array(valence_std).reshape(-1, )
        total_mean   = np.array(total_mean).reshape(-1, )
        total_std    = np.array(total_std).reshape(-1, )

        plt.figure(1)
        plt.errorbar(x, arousal_mean, yerr=arousal_std, ecolor=ecolor, mec=ecolor, mfc=ecolor, ms=10, capsize=4, fmt='s')
        plt.title('Arousal')
        plt.ylabel('Error')
        plt.xticks([1, 3, 5, 7, 9], modalities)

        plt.figure(2)
        plt.errorbar(x, valence_mean, yerr=valence_std, ecolor=ecolor, mec=ecolor, mfc=ecolor, ms=10, capsize=4, fmt='s')
        plt.title('Valence')
        plt.ylabel('Error')
        plt.xticks([1, 3, 5, 7, 9], modalities)

        if dataset == 'Test':
            plt.figure(3)
            plt.errorbar(x, total_mean, yerr=total_std, ecolor=ecolor, mec=ecolor, mfc=ecolor, ms=10, capsize=4, fmt='s')
            plt.title('Total')
            plt.ylabel('Error')
            plt.xticks(x, modalities)
    
    plt.figure(1)
    plt.savefig(Path(__file__).parents[1] / 'results/Regression/Arousal_error.png')

    plt.figure(2)
    plt.savefig(Path(__file__).parents[1] / 'results/Regression/Valence_error.png')
    
    plt.figure(3)
    plt.savefig(Path(__file__).parents[1] / 'results/Regression/Total_error.png')


#FIXME: Dont use this code
##########################################################################
# Segregate data based on the increasing values of valence - Category 1 #
##########################################################################
# 11 images correpond to a single event of 15 s length and sliding window of 0.5 s
# k1, k2, ks, ke = 3, 3, 0, 8   # for considering first 3 samples as our dataset 
# k1, k2, ks, ke = 3, 3, 8, 8   # for considering last 3 samples as our dataset 
k1, k2, ks, ke = 11, 11, 0, 0 # for considering all 11 samples as our dataset

with skip_run('skip', 'Split features into training and testing sets based on increasing valence levels') as check, check():
    # oasis_filepath = Path(__file__).parents[1] / config['oasis_path']
    # row numbers 400 and 403 have same values and 758 and 838 have same values
    # here we are removing the rows 403 and 838 from consideration
    # df = pd.read_csv(oasis_filepath, delimiter=',', skiprows=[403, 838], usecols=[6,7])

    data = dd.io.load(config['hri_feats'])
    # data = dd.io.load(config['hri_feats_first_samp'])

    Data = collections.defaultdict(dict)
    for subject in data.keys():
        hri_ecg, hri_emg, hri_gsr, hri_ppg, hri_rsp, hri_labels = [], [], [], [], [], []
        # combine the data of two events
        for event in data[subject]:
            hri_ecg.append(data[subject][event]['ECG'])
            hri_emg.append(data[subject][event]['EMG'])
            hri_gsr.append(data[subject][event]['GSR'])
            hri_ppg.append(data[subject][event]['PPG'])
            # hri_rsp.append(data[subject][event]['RSP'])
            hri_labels.append(np.round(data[subject][event]['labels'], 2)) 
    
        hri_ecg    = np.concatenate(hri_ecg, axis=0)
        hri_emg    = np.concatenate(hri_emg, axis=0)
        hri_gsr    = np.concatenate(hri_gsr, axis=0)
        hri_ppg    = np.concatenate(hri_ppg, axis=0)
        # hri_rsp    = np.concatenate(hri_rsp, axis=0)
        hri_labels = np.concatenate(hri_labels, axis=0)

        # sort according to valence increasing order 
        sorted_ind = np.argsort(hri_labels[:, 1])

        hri_ecg         = hri_ecg[sorted_ind, :, :]   
        hri_emg         = hri_emg[sorted_ind, :, :]
        hri_gsr         = hri_gsr[sorted_ind, :, :]
        hri_ppg         = hri_ppg[sorted_ind, :, :]
        # hri_rsp         = hri_rsp[sorted_ind, :, :]
        hri_labels      = hri_labels[sorted_ind, :]

        # plt.cla()
        N = hri_labels.shape[0]
        ind = np.arange(N)

        train_ind, test_ind = [], []
        for i in range(0, N, (k1+k2+2*np.max([ks,ke]))):
            train_ind.append(ind[i+ks:i+k1+ks])
            test_ind.append(ind[i+k1+ks+ke:i+k1+k2+ks+ke])

        train_ind = np.concatenate(train_ind, axis=0)
        test_ind  = np.concatenate(test_ind, axis=0)

        train_data = {'ECG':hri_ecg[train_ind, :, :],
                      'EMG':hri_emg[train_ind, :, :],
                      'GSR':hri_gsr[train_ind, :, :],
                      'PPG':hri_ppg[train_ind, :, :],
                    #   'RSP':hri_rsp[train_ind, :, :],
                      'labels':hri_labels[train_ind, :]}
    
        test_data  = {'ECG':hri_ecg[test_ind, :, :],
                      'EMG':hri_emg[test_ind, :, :],
                      'GSR':hri_gsr[test_ind, :, :],
                      'PPG':hri_ppg[test_ind, :, :],
                    #   'RSP':hri_rsp[test_ind, :, :],
                      'labels':hri_labels[test_ind, :]}

        # plt.plot(train_data['labels'][:, 1], train_data['labels'][:, 0], 'b.')
        # plt.plot(test_data['labels'][:, 1], test_data['labels'][:, 0], 'r.')
        # plt.pause(1)

        Data[subject] = {'train': train_data,
                        'test' : test_data}
    
    save_path = str(Path(__file__).parents[1] / config['hri_feats_category1'])
    dd.io.save(save_path, Data)

with skip_run('skip', 'Split ECG-SSL features into training and testing sets based on increasing valence levels') as check, check():
    data = dd.io.load(config['hri_ECG_SSL_feats'])
    # data = dd.io.load(config['hri_ECG_SSL_feats_first_samp'])

    Data = collections.defaultdict(dict)
    for subject in data.keys():
        hri_ecg, hri_labels = [], []
        # combine the data of two events
        for event in data[subject]:
            hri_ecg.append(data[subject][event]['ECG'])
            hri_labels.append(np.round(data[subject][event]['labels'], 2)) 
    
        hri_ecg    = np.concatenate(hri_ecg, axis=0)
        hri_labels = np.concatenate(hri_labels, axis=0)

        # sort according to valence increasing order 
        sorted_ind = np.argsort(hri_labels[:, 1])

        hri_ecg         = hri_ecg[sorted_ind, :]   
        hri_labels      = hri_labels[sorted_ind, :]

        # plt.cla()
        N = hri_labels.shape[0]
        ind = np.arange(N)

        train_ind, test_ind = [], []
        for i in range(0, N, (k1+k2+2*np.max([ks,ke]))):
            train_ind.append(ind[i+ks:i+k1+ks])
            test_ind.append(ind[i+k1+ks+ke:i+k1+k2+ks+ke])

        train_ind = np.concatenate(train_ind, axis=0)
        test_ind  = np.concatenate(test_ind, axis=0)

        train_data = {'ECG':hri_ecg[train_ind, :],
                      'labels':hri_labels[train_ind, :]}
    
        test_data  = {'ECG':hri_ecg[test_ind, :],
                      'labels':hri_labels[test_ind, :]}

        # plt.plot(train_data['labels'][:, 1], train_data['labels'][:, 0], 'b.')
        # plt.plot(test_data['labels'][:, 1], test_data['labels'][:, 0], 'r.')
        # plt.pause(1)

        Data[subject] = {'train': train_data,
                        'test' : test_data}
    
    save_path = str(Path(__file__).parents[1] / config['hri_ECG_SSL_feats_category1'])
    dd.io.save(save_path, Data)


# 1) Emotion recognition for each subject and category 1 data using each modality
with skip_run('skip', 'train Regressor on the ECG-SSL features') as check, check():
    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/category_regression/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/category_regression/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/category_regression/')

    data = dd.io.load(config['hri_ECG_SSL_feats_category1'])

    for subject in data.keys():
        print('---------', subject, '---------')
        train_data = data[subject]['train']
        test_data  = data[subject]['test']

        modality = 'ECG-SSL'
        # Linear regression and SVR are not performing well in estimating the valence-arousal levels
        # regressor = MultiOutputRegressor(Ridge()) 
        # regressor = MultiOutputRegressor(SVR())

        # Until now RandomForest has done best job in recognizing the valence-arousal levels
        regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))
        # regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=30))
        # regressor = MultiOutputRegressor(xgb.XGBRegressor())
        # regressor = MultiOutputRegressor(xgb.XGBRFRegressor(n_estimators=100, max_depth=30))

        train_dataset = {'features': train_data['ECG'],
                        'labels': train_data['labels']}
                        
        test_dataset  = {'features': test_data['ECG'],
                        'labels': test_data['labels']}
        
        model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/category_regression/' + subject + '_' + modality + '.pkl'
        pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/category_regression/'+ subject + '_' + modality + '.png'
        results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/category_regression/' + subject + '_' + modality + '.csv'

        train_test_regression_model(modality, train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)
    # plt.close('all')

with skip_run('skip', 'train Regressor using hand-crafted features for each modalities') as check, check():
    
    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/category_regression/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/category_regression/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/category_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/category_regression/')

    data = dd.io.load(config['hri_feats_category1'])

    for subject in data.keys():
        plt.close('all')
        print('---------', subject, '---------')
        train_data = data[subject]['train']
        test_data  = data[subject]['test']

        # Linear regression and SVR are not performing well in estimating the valence-arousal levels
        # regressor = MultiOutputRegressor(Ridge()) 
        # regressor = MultiOutputRegressor(SVR())

        for modality in config['hri']['ch_names']:
            
            if modality in train_data.keys():
                # Until now RandomForest has done best job in recognizing the valence-arousal levels
                regressor = (RandomForestRegressor(n_estimators=100, max_depth=30))
                
                model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/category_regression/' + subject + '_' + modality + '.pkl'
                pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/category_regression/' + subject + '_' + modality + '.png'
                results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/category_regression/' + subject + '_' + modality + '.csv'

                # Creating dataset after taking care of additional dimension at axis=1
                train_dataset = {'features': train_data[modality].reshape(train_data[modality].shape[0], -1),
                                'labels': train_data['labels']} 
                test_dataset  = {'features': test_data[modality].reshape(test_data[modality].shape[0], -1),
                                'labels': test_data['labels']}

                train_test_regression_model(modality, train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)


# 2) Multimodal emotion recognition for each subject using category 1 data and Random Forest Regression
with skip_run('skip', 'RF - Multi-modal emotion recognition using ECG, EMG, GSR, RSP features') as check, check():
    
    # --------balance the training data ----------
    balance_train_data = False
    # --------------------------------------------
    
    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/multimodal_regression/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/multimodal_regression/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/multimodal_regression/')

    data = dd.io.load(config['hri_feats_category1'])

    for subject in data.keys():
        print('---------', subject, '---------')
        train_data = data[subject]['train']
        test_data  = data[subject]['test']

        # Linear regression and SVR are not performing well in estimating the valence-arousal levels
        # regressor = MultiOutputRegressor(Ridge()) 
        # regressor = MultiOutputRegressor(SVR())
        
        # Until now RandomForest has done best job in recognizing the valence-arousal levels
        regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))
        
        model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/multimodal_regression/' + subject + '.pkl'
        pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/multimodal_regression/' + subject + '.png'
        results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/multimodal_regression/' + subject + '.csv'

        train_feats, test_feats = [], []
        for modality in config['hri']['ch_names']:
            if modality in train_data.keys():
                train_feats.append(train_data[modality].reshape(train_data[modality].shape[0], -1))
                test_feats.append(test_data[modality].reshape(test_data[modality].shape[0], -1))
        
        train_feats = np.concatenate(train_feats, axis=1)
        test_feats = np.concatenate(test_feats, axis=1)

        print('Train features size: ', train_feats.shape)
        print('Test features size: ', test_feats.shape)

        if balance_train_data:
            bal_ind = utils.balance_labels(train_data['labels'])
        else:
            bal_ind = np.arange(train_data['labels'].shape[0])

        # Creating dataset after taking care of additional dimension at axis=1
        train_dataset = {'features': train_feats[bal_ind, :],
                        'labels': train_data['labels'][bal_ind, :]} 
        test_dataset  = {'features': test_feats,
                        'labels': test_data['labels']}

        train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)

with skip_run('skip', 'RF - Multi-modal emotion recognition using ECG-SSL, EMG, GSR, RSP features') as check, check():
    # --------balance the training data ----------
    balance_train_data = False
    # --------------------------------------------

    if ~os.path.exists(Path(__file__).parents[1] / 'models/Regression_models/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'models/Regression_models/multimodal_regression/')
        
    if ~os.path.exists(Path(__file__).parents[1] / 'docs/Images/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'docs/Images/multimodal_regression/')
    
    if ~os.path.exists(Path(__file__).parents[1] / 'results/Regression/multimodal_regression/'):
        utils.makedirs(Path(__file__).parents[1] / 'results/Regression/multimodal_regression/')

    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats_category1'])
    data = dd.io.load(config['hri_feats_category1'])

    for subject in data.keys():
        print('---------', subject, '---------')
        train_data = data[subject]['train']
        test_data  = data[subject]['test']

        train_ECG_SSL = data_ECG_SSL[subject]['train']
        test_ECG_SSL = data_ECG_SSL[subject]['test']

        # Linear regression and SVR are not performing well in estimating the valence-arousal levels
        # regressor = MultiOutputRegressor(Ridge()) 
        # regressor = MultiOutputRegressor(SVR())
        
        # Until now RandomForest has done best job in recognizing the valence-arousal levels
        regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))
        
        model_save_path = str(Path(__file__).parents[1]) + '/models/Regression_models/multimodal_regression/' + subject + '.pkl'
        pic_save_path   = str(Path(__file__).parents[1]) + '/docs/Images/multimodal_regression/' + subject + '.png'
        results_save_path = str(Path(__file__).parents[1]) + '/results/Regression/multimodal_regression/' + subject + '.csv'

        train_feats, test_feats = [], []
        train_feats.append(train_ECG_SSL['ECG'])
        test_feats.append(test_ECG_SSL['ECG'])

        for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
            if modality in train_data.keys(): 
                train_feats.append(train_data[modality].reshape(train_data[modality].shape[0], -1))
                test_feats.append(test_data[modality].reshape(test_data[modality].shape[0], -1))
        
        train_feats = np.concatenate(train_feats, axis=1)
        test_feats = np.concatenate(test_feats, axis=1)

        print('Train features size: ', train_feats.shape)
        print('Test features size: ', test_feats.shape)

        if balance_train_data:
            bal_ind = utils.balance_labels(train_data['labels'])
        else:
            bal_ind = np.arange(train_data['labels'].shape[0])

        # Creating dataset after taking care of additional dimension at axis=1
        train_dataset = {'features': train_feats[bal_ind, :],
                        'labels': train_data['labels'][bal_ind, :]} 
        test_dataset  = {'features': test_feats,
                        'labels': test_data['labels']}

        train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=model_save_path, pic_save_path=pic_save_path, results_save_path=results_save_path)

# 3) Multimodal emotion recognition for each subject using category 1 data and DNN
with skip_run('skip', 'DNN - Multi-modal emotion recogition using ECG-SSL, EMG, GSR, RSP features') as check, check():
    # --------balance the training data ----------
    balance_train_data = True
    # --------------------------------------------

    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats_category1'])
    data = dd.io.load(config['hri_feats_category1'])

    for subject in data.keys():
        print('---------', subject, '---------')
        train_data = data[subject]['train']
        test_data  = data[subject]['test']

        train_ECG_SSL = data_ECG_SSL[subject]['train']
        test_ECG_SSL = data_ECG_SSL[subject]['test']

        train_feats, test_feats = [], []
        train_feats.append(train_ECG_SSL['ECG'])
        test_feats.append(test_ECG_SSL['ECG'])

        for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
            if modality in train_data.keys():
                # normalize the features and then append them
                scaler = StandardScaler()
                scaler.fit(train_data[modality].reshape(train_data[modality].shape[0], -1))

                train_feats.append(scaler.transform(train_data[modality].reshape(train_data[modality].shape[0], -1)))
                test_feats.append(scaler.transform(test_data[modality].reshape(test_data[modality].shape[0], -1)))
        
        train_feats = np.concatenate(train_feats, axis=1)
        test_feats = np.concatenate(test_feats, axis=1)

        print('Train features size: ', train_feats.shape)
        print('Test features size: ', test_feats.shape)

        # if balance_train_data:
        #     bal_ind = utils.balance_labels(train_data['labels'])
        # else:
        #     bal_ind = np.arange(train_data['labels'].shape[0])

        # Creating dataset after taking care of additional dimension at axis=1
        train_dataset = {'features': train_feats,
                        'labels': train_data['labels']} 
        test_dataset  = {'features': test_feats,
                        'labels': test_data['labels']}

        batch_size = 128
        window_size = config['freq'] * config['window_size']

        train_data = MultiFeatDataset(train_dataset, balance_data=False)
        test_data  = MultiFeatDataset(test_dataset, balance_data=False)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Train the model on: {}'.format(device))

        net = EmotionNet(num_feats=train_dataset['features'].shape[1], device=device, config=config)
        net.to(device)

        # see if an exponential decay learning rate scheduler is required
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        scheduler = None
        net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=500)

# 4) Test how small the training dataset could be after which the dataset diverges
with skip_run('skip', 'Split features into training and testing sets based on increasing valence levels') as check, check():
    data = dd.io.load(config['hri_feats'])
    data_SSL = dd.io.load(config['hri_ECG_SSL_feats'])

    Features, Labels = [], []
    for sub in data:
        for event in data[sub]:
            features, labels = [], []
            labels = data[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in ['EMG', 'GSR', 'PPG', 'RSP']:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))
            
            features = np.concatenate(features, axis=1)

            Features.append(features)
            Labels.append(labels)

    Features = np.concatenate(Features, axis=0)
    Labels   = np.concatenate(Labels, axis=0)


#FIXME: Dont use this code
##########################################################################
# Segregated data based on the Image categories from Mike - Category 2 #
##########################################################################
with skip_run('skip', 'Segregated data based on the Image categories') as check, check():
    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats_category1'])
    data = dd.io.load(config['hri_feats_category1'])

    for sub in data.keys():
        print('------------------- ', sub , ' -------------------')
        for dataset in ['train', 'test']:
            features, labels = [], []
            features.append(data_ECG_SSL[sub][dataset]['ECG'])
            labels = data_ECG_SSL[sub][dataset]['labels']
            ind = np.arange(labels.shape[0])

            for modality in ['ECG', 'EMG', 'GSR', 'PPG', 'RSP']:
                if modality in data[sub][dataset].keys():
                    features.append(data[sub][dataset][modality].reshape(data[sub][dataset][modality].shape[0], -1))

            features = np.concatenate(features, axis=1)
            
            if dataset == 'train':
                # Arousal-Valence limits defined by Mike
                # neutral_ind  = ind[((labels[:, 0] <= 2.5) & (labels[:, 0] >=1) & (labels[:, 1] <=4.5) & (labels[:, 1] >=3.5))]
                # high_val_ind = ind[((labels[:, 0] >= 4) & (labels[:, 1] >= 5.1))]
                # low_val_ind  = ind[((labels[:, 0] >= 4) & (labels[:, 1] <= 2.9))]

                # neutral_ind = neutral_ind[1:-1:3]
                # ind = np.concatenate((neutral_ind, low_val_ind, high_val_ind), axis=0)

                train_dataset = {'features': features[ind, :],
                                    'labels': labels[ind, :]}
            else:

                test_dataset  = {'features': features[ind, :],
                                'labels': labels[ind, :]}

        regressor = (RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=100, warm_start=True, oob_score=True))
        # regressor = MultiOutputRegressor(AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=30),
                                                        #    n_estimators=50, 
                                                        #    learning_rate=1, 
                                                        #    loss='square'))

        train_test_regression_model('', train_dataset, test_dataset, regressor, clean_features=True, model_save_path=[], pic_save_path=[], results_save_path=[])
            

##############################################################
# learn dynamics model from windows spanning multiple images #
##############################################################
with skip_run('skip', 'Learn dynamics model from windows spanning multiple images') as check, check():
    train_model = True
    # number of LSTM cells; if 1, use regular DNN
    # sequence_length = 11
    sequence_length = 1

    # load data: SSL
    # data = dd.io.load(config['hri_feats_transition'])
    data_ECG_SSL = dd.io.load(config['hri_ECG_SSL_feats_transition'])

    Features, Labels = [], []

    event_end_idx = []
    target_subs = ['S5']
    #modalities = ['EMG', 'GSR', 'PPG', 'RSP']
    modalities = []
    #for sub in data:
    for sub in target_subs:
        print('sub:', sub)
        for event in data_ECG_SSL[sub]:
            features, labels = [], []
            labels = data_ECG_SSL[sub][event]['labels']

            # use ECG-SSL, EMG, GSR, PPG for regression
            features.append(data_ECG_SSL[sub][event]['ECG'].reshape(data_ECG_SSL[sub][event]['ECG'].shape[0], -1))
            for modality in modalities:
                if modality in data[sub][event].keys():
                    # normalize the features and then append them for DNN, it does not matter for Random Forest
                    scaler = StandardScaler()
                    features.append(scaler.fit_transform(data[sub][event][modality].reshape(data[sub][event][modality].shape[0], -1)))

            features = np.concatenate(features, axis=1)

            Features.append(features)
            Labels.append(labels)
            if len(event_end_idx) > 0:
                event_end_idx.append(event_end_idx[-1] + labels.shape[0])
            else:
                event_end_idx.append(labels.shape[0])
            print('event:', event, ', labels.shape:', labels.shape, ', event_end_idx:', event_end_idx[-1])

    Features = np.concatenate(Features, axis=0)
    Labels   = np.concatenate(Labels, axis=0)

    print('Features shape (all):', Features.shape)
    print('Labels shape (all):', Labels.shape)

    # categorize into neutral, positive, negative images
    label_first_idx = [0]
    label_list = [Labels[0, :]]
    for i in range(Labels.shape[0]):
        if i > 0:
            if Labels[i-1, 0] != Labels[i, 0] or Labels[i-1, 1] != Labels[i, 1]:
                label_first_idx.append(i)
                label_list.append(Labels[i, :])

    categories = []
    pos_image_ids = []
    neg_image_ids = []
    for i, label in enumerate(label_list):
        if label[0] <= 2.5:
            if label[1] >= 2.5 and label[1] <= 4.5:
                # print(i, label, ' neutral')
                categories.append(0)

            elif label[1] > 4.5:
                # print(i, label, ' positive')
                categories.append(1)
                pos_image_ids.append(i)

            else:
                # print(i, label, ' negative')
                categories.append(-1)
                neg_image_ids.append(i)

        else:
            if label[1] < 3.5:
                # print(i, label, ' negative')
                categories.append(-1)
                neg_image_ids.append(i)

            else:
                # print(i, label, ' positive')
                categories.append(1)
                pos_image_ids.append(i)

        # check if neutral images come every other frame
        if categories[-1] == 0:
            if len(categories) > 1 and categories[-2] == 0:
                print('---- order error ----')

            elif len(categories) > 2 and categories[-3] != 0:
                print('---- order error ----')

        else:
            if len(categories) > 1 and categories[-2] != 0:
                print('---- order error ----')

            elif len(categories) > 2 and categories[-3] == 0:
                print('---- order error ----')

    per_train_data, RF_ars_mean, RF_val_mean, NN_ars_mean, NN_val_mean = [], [], [], [], []
    RF_ars_err, RF_val_err, NN_ars_err, NN_val_err = [], [], [], []

    # use event1 for training and event2 for testing
    train_ind = range(0, sequence_length * (event_end_idx[0] // sequence_length))
    test_ind = range(event_end_idx[0], event_end_idx[0] + sequence_length * ((event_end_idx[1] - event_end_idx[0]) // sequence_length))

    train_dataset = {'features': Features[train_ind, :],
                     'labels': Labels[train_ind, :]}

    test_dataset = {'features': Features[test_ind, :],
                    'labels': Labels[test_ind, :]}

    print('train feature shape:', train_dataset['features'].shape, ', label shape:', train_dataset['labels'].shape)
    print('test feature shape:', test_dataset['features'].shape, ', label shape:', test_dataset['labels'].shape)

    train_data = MultiFeatDataset(train_dataset, balance_data=False)
    test_data  = MultiFeatDataset(test_dataset, balance_data=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # regular NN
    if sequence_length == 1:
        batch_size = 128
        net = EmotionNet(num_feats=train_dataset['features'].shape[1], device=device, config=config)
    # LSTM
    else:
        batch_size = 8
        net = EmotionNetLSTM(num_feats=train_dataset['features'].shape[1], seq_len=sequence_length, device=device, config=config)

    net.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=sequence_length*batch_size, shuffle=False)
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=sequence_length*batch_size, shuffle=False)

    # see if an exponential decay learning rate scheduler is required
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = None
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if train_model:
        print('Train the model on: {}'.format(device))
        NN_trn_ars, NN_trn_val, NN_tst_ars, NN_tst_val = net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=scheduler, batch_size=batch_size, epochs=2000)
    else:
        torch_file = 'MultiModal_model_' + str(NN_model_dic[str(step_size)]) + '/net_500.pth'

        checkpoint = torch.load(config['torch']['EMOTION_models'] + torch_file, map_location=device)
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('DNN')

        trn_results = net.validate_model(train_dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=ax[0])
        tst_results = net.validate_model(test_dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=ax[1])

        NN_trn_ars = trn_results['Arousal']
        NN_trn_val = trn_results['Valence']
        NN_tst_ars = tst_results['Arousal']
        NN_tst_val = tst_results['Valence']

    print(NN_trn_ars.shape, NN_trn_val.shape, NN_tst_ars.shape, NN_tst_val.shape)

    ax1[0, 1].hist(NN_tst_ars, bins=n_bins)
    ax1[0, 1].set_title('Arousal-DNN ({})'.format(step_size))
    ax1[0, 1].set_xlim([0, 4])
    ax1[0, 1].set_ylim([0, 6000])

    ax1[1, 1].hist(NN_tst_val, bins=n_bins)
    ax1[1, 1].set_title('Valence-DNN ({})'.format(step_size))
    ax1[1, 1].set_xlim([0, 4])
    ax1[1, 1].set_ylim([0, 6000])
    fig1.suptitle('Error histogram')

    # ax2[0, 1].hist(NN_trn_val, bins=n_bins)
    # ax2[0, 1].set_title('DNN-Train')

    # ax2[1, 1].hist(NN_tst_val, bins=n_bins)
    # ax2[1, 1].set_title('DNN-Test')
    # fig2.suptitle('Valence absolute error')

    per_train_data.append((train_categories.shape[0] * 100 / num_cat))
    NN_ars_mean.append(np.mean(NN_tst_ars))
    NN_val_mean.append(np.mean(NN_tst_val))

    # RF_ars_err.append(np.array([np.mean(RF_tst_ars) - np.min(RF_tst_ars), np.max(RF_tst_ars) - np.mean(RF_tst_ars)]).reshape(2, 1))
    # RF_val_err.append(np.array([np.mean(RF_tst_val) - np.min(RF_tst_val), np.max(RF_tst_val) - np.mean(RF_tst_val)]).reshape(2, 1))
    # NN_ars_err.append(np.array([np.mean(NN_tst_ars) - np.min(NN_tst_ars), np.max(NN_tst_ars) - np.mean(NN_tst_ars)]).reshape(2, 1))
    # NN_val_err.append(np.array([np.mean(NN_tst_val) - np.min(NN_tst_val), np.max(NN_tst_val) - np.mean(NN_tst_val)]).reshape(2, 1))

    NN_ars_err.append(np.std(NN_tst_ars))
    NN_val_err.append(np.std(NN_tst_val))

    # NN_ars_err = np.concatenate(NN_ars_err, axis=1)
    # NN_val_err = np.concatenate(NN_val_err, axis=1)
    per_train_data = np.array(per_train_data)

    fig, ax = plt.subplots(1, 2)
    ax[0].errorbar(per_train_data, NN_ars_mean, yerr=NN_ars_err, ecolor='k', mec='k', mfc='k', ms=10, capsize=4, fmt='s', label='DNN')
    ax[0].set_title('Arousal test error (DNN)')
    ax[0].set_ylabel('Mean Absolute Error')
    ax[0].set_xlabel('Percentage training data')
    ax[0].set_ylim([0, 2.5])
    ax[0].set_xlim([0, 60])

    ax[1].errorbar(per_train_data, NN_val_mean, yerr=NN_val_err, ecolor='k', mec='k', mfc='k', ms=10, capsize=4, fmt='s', label='DNN')
    ax[1].set_title('Valence test error (DNN)')
    # ax[1].set_ylabel('Mean Absolute Error')
    ax[1].set_xlabel('Percentage training data')
    ax[1].set_ylim([0, 2.5])
    ax[1].set_xlim([0, 60])

plt.show()
