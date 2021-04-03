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

import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


from NN_models import SelfSupervisedNetFeats, EcgNet, EmotionNet
from NN_datasets import EcgDataset, EcgFeatDataset, MultiFeatDataset

from data_preprocessing import read_raw_hri_dataset, read_individual_hri_dataset
from feature_extraction import extract_all_features
from Regression_models import train_regression_model
import signal_transformation as sgtf
import utils
from utils import skip_run

config = yaml.load(open(Path(__file__).parents[1] / 'config.yml'), Loader=yaml.SafeLoader)
window_len = config['freq'] * config['window_size']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128

#################################   
# 1) Read S1's calibration data - obtain the mean and std for each signal to Standardize the downstream task
#################################
calibration_data, scaler_dict = read_individual_hri_dataset("data/Hug_exp/S1/calibration", "data/Hug_exp/S1_calib.h5", config['hri']['percent_overlap'], config, save=True, standardize=True, input_scaler=None, calib=True)
# plt.plot(calibration_data['S1']['event']['labels'][:, 1], calibration_data['S1']['event']['labels'][:, 0], '.')

# Extract SSL_features
if  not os.path.isfile("data/Hug_exp/S1_calib_feats.h5"):
    ckpt_file  = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    ECG_data = EcgDataset("data/Hug_exp/S1_calib.h5", window_len, data_group=str('/data/event'))
    SSL_dataloader = torch.utils.data.DataLoader(ECG_data, batch_size=batch_size, shuffle=False)
    SSL_features, labels = net.return_SSL_feats(SSL_dataloader)

    Multi_features = extract_all_features(calibration_data['event'], config)
    Multi_features['ECG_SSL'] = SSL_features

    dd.io.save("data/Hug_exp/S1_calib_feats.h5", Multi_features)

balance_features = False
features = dd.io.load("data/Hug_exp/S1_calib_feats.h5")

train_feats = []
feat_scaler = {}
for modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
    if modality in features.keys():
        feat_scaler[modality] =  StandardScaler().fit(features[modality].reshape(features[modality].shape[0], -1))

        if modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
            transformed_features = feat_scaler[modality].fit_transform(features[modality].reshape(features[modality].shape[0], -1))
        else:
            transformed_features = features[modality].reshape(features[modality].shape[0], -1)

        train_feats.append(features[modality].reshape(features[modality].shape[0], -1))
        

train_feats = np.concatenate(train_feats, axis=1)

if balance_features:
    bal_ind = utils.balance_labels(features['labels'])
else:
    bal_ind = np.arange(features['labels'].shape[0])

# Creating dataset after taking care of additional dimension at axis=1
train_dataset = {'features': train_feats[bal_ind, :],
                'labels': features['labels'][bal_ind, :]} 

#############################
# Read S1's experiment Human robot experiment data 
#############################
exp_data, _ = read_individual_hri_dataset("data/Hug_exp/S1/exp4", "data/Hug_exp/S1_exp.h5", config['hri']['percent_overlap'], config, save=True, standardize=False, input_scaler=scaler_dict)

# Extract SSL_features
if  not os.path.isfile("data/Hug_exp/S1_exp_feats.h5"):
    ckpt_file  = 'model2_1/net_200.pth'
    checkpoint = torch.load(config['torch']['SSL_models'] + ckpt_file, map_location=device)
    net = SelfSupervisedNetFeats(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

    ECG_data = EcgDataset("data/Hug_exp/S1_exp.h5", window_len, data_group=str('/data/event'))
    SSL_dataloader = torch.utils.data.DataLoader(ECG_data, batch_size=batch_size, shuffle=False)
    SSL_features, labels = net.return_SSL_feats(SSL_dataloader)

    # ECG is only recorded for sometime, remove the additional data from other modalities to match the size 
    ind_size = exp_data['event']['ECG'].shape[0]

    for modality in ['ECG', 'EMG', 'GSR', 'PPG', 'RSP']:
        exp_data['event'][modality] = exp_data['event'][modality][:ind_size, :, :]

    Multi_features = extract_all_features(exp_data['event'], config)
    Multi_features['ECG_SSL'] = SSL_features

    dd.io.save("data/Hug_exp/S1_exp_feats.h5", Multi_features)

balance_features = False
features_exp = dd.io.load("data/Hug_exp/S1_exp_feats.h5")

test_feats = []
for modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
    if modality in features_exp.keys(): 
        if modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
            transformed_features = feat_scaler[modality].fit_transform(features_exp[modality].reshape(features_exp[modality].shape[0], -1))
        else:
            transformed_features = features_exp[modality].reshape(features_exp[modality].shape[0], -1)
        
        test_feats.append(transformed_features)

test_feats = np.concatenate(test_feats, axis=1)

if balance_features:
    bal_ind = utils.balance_labels(features_exp['ECG_SSL'])
else:
    bal_ind = np.arange(features_exp['ECG_SSL'].shape[0])

# Creating dataset after taking care of additional dimension at axis=1
test_dataset = {'features': test_feats[bal_ind, :], 'labels': []} 

# Handle the nans and zeros in data
# test_dataset['features'] = np.nan_to_num(test_dataset['features'], nan=0.0, posinf=0.0, neginf=0.0)

if not os.path.isfile("data/Hug_exp/RF_S1.pkl"):
    regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))
    regressor = train_regression_model('', train_dataset, regressor, clean_features=True, model_save_path="data/Hug_exp/RF_S1.pkl")
else:
    regressor = pickle.load(open("data/Hug_exp/RF_S1.pkl", 'rb'))


RF_pred  = regressor.predict(test_dataset['features'])     
time = np.arange(RF_pred.shape[0])
fig, ax = plt.subplots(3, 1)
ax[0].plot(time, RF_pred[:, 1], 'r.', label='RF Predictions')
ax[0].set_ylabel('Valence')
ax[0].set_ylim([1, 6])

ax[1].plot(time, RF_pred[:, 0], 'r.', label='RF Predictions')
ax[1].set_ylabel('Arousal')
ax[1].set_ylim([1, 6])

ax[2].plot(RF_pred[:, 1], RF_pred[:, 0], 'r.', label='RF Predictions')
ax[2].set_xlim([1, 6])
ax[2].set_ylim([1, 6])
fig.suptitle('Multimodal RF predictions')

# Predictions using EmotionNet
ckpt_file  = 'MultiModal_320_feats/net_500.pth'
checkpoint = torch.load(config['torch']['EMOTION_models'] + ckpt_file, map_location=device)
net = EmotionNet(num_feats=test_dataset['features'].shape[1], device=device, config=config).to(device)

multifeat_dataset = MultiFeatDataset(test_dataset)
multifeat_dataloader = torch.utils.data.DataLoader(multifeat_dataset, batch_size=batch_size, shuffle=False)
   
NN_pred = net.predict(multifeat_dataloader, checkpoint)

fig, ax = plt.subplots(3, 1)
ax[0].plot(time, NN_pred[:, 1], 'r.', label='NN Predictions')
ax[0].set_ylabel('Valence')
# ax[0].set_ylim([1, 6])

ax[1].plot(time, NN_pred[:, 0], 'r.', label='NN Predictions')
ax[1].set_ylabel('Arousal')
# ax[1].set_ylim([1, 6])

ax[2].plot(NN_pred[:, 1], NN_pred[:, 0], 'r.', label='NN Predictions')
ax[2].set_xlim([1, 6])
ax[2].set_ylim([1, 6])

fig.suptitle('Multimodal NN predictions')



# # Predictions based on oinly ECG-SSL features
# ECG_SSL_feats = features_exp['ECG_SSL'].reshape(features_exp['ECG_SSL'].shape[0], -1)

# # Creating dataset after taking care of additional dimension at axis=1
# SSL_dataset = {'features': ECG_SSL_feats[bal_ind, :], 'labels': []} 

# # test using previously trained model EmotionNet on pooled data
# ckpt_file  = 'model_EcgNet/net_500.pth'
# checkpoint = torch.load(config['torch']['EMOTION_models'] + ckpt_file, map_location=device)
# net = EcgNet(load_model=True, checkpoint=checkpoint, device=device, config=config).to(device)

# SSL_dataset = MultiFeatDataset(SSL_dataset)
# SSL_dataloader = torch.utils.data.DataLoader(SSL_dataset, batch_size=batch_size, shuffle=False)
   
# ECG_SSL_pred = net.predict(SSL_dataloader, checkpoint)


# fig, ax = plt.subplots(2, 1)
# ax[0].plot(time, ECG_SSL_pred[:, 1], 'r.', label='ECG_SSL Predictions')
# ax[0].set_ylabel('Valence')
# # ax[0].set_ylim([1, 6])

# ax[1].plot(time, ECG_SSL_pred[:, 0], 'r.', label='ECG_SSL Predictions')
# ax[1].set_ylabel('Arousal')
# # ax[1].set_ylim([1, 6])
# fig.suptitle('Multimodal ECG_SSL predictions')

plt.show()
