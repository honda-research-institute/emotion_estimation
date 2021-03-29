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
# 1) Read Mike's calibration data
#################################
calibration_data = read_individual_hri_dataset("data/Hug_exp/S1/calibration", "data/Hug_exp/S1_calib.h5", config['hri']['percent_overlap'], config, save=True, standardize=False, calib=True)
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
for modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
    if modality in features.keys(): 
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
# Read Mike's experiment data
#############################
exp_data = read_individual_hri_dataset("data/Hug_exp/S1/exp4", "data/Hug_exp/S1_exp.h5", config['hri']['percent_overlap'], config, save=True, standardize=False)

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
        print(exp_data['event'][modality].shape)

    Multi_features = extract_all_features(exp_data['event'], config)
    Multi_features['ECG_SSL'] = SSL_features

    dd.io.save("data/Hug_exp/S1_exp_feats.h5", Multi_features)

balance_features = False
features_exp = dd.io.load("data/Hug_exp/S1_exp_feats.h5")

test_feats = []
for modality in ['ECG_SSL', 'EMG', 'GSR', 'PPG', 'RSP']:
    if modality in features_exp.keys(): 
        test_feats.append(features_exp[modality].reshape(features_exp[modality].shape[0], -1))

test_feats = np.concatenate(test_feats, axis=1)

if balance_features:
    bal_ind = utils.balance_labels(features_exp['EMG'])
else:
    bal_ind = np.arange(features_exp['EMG'].shape[0])

# Creating dataset after taking care of additional dimension at axis=1
test_dataset = {'features': test_feats[bal_ind, :]} 
# Handle the nans and zeros in data
test_dataset['features'] = np.nan_to_num(test_dataset['features'], nan=0.0, posinf=0.0, neginf=0.0)

if not os.path.isfile("data/Hug_exp/RF_S1.pkl"):
    regressor = (RandomForestRegressor(n_estimators=1000, max_depth=30))
    regressor = train_regression_model('', train_dataset, regressor, clean_features=True, model_save_path=[])
    pickle.dump(regressor, open("data/Hug_exp/RF_S1.pkl", 'wb'))
else:
    regressor = pickle.load(open("data/Hug_exp/RF_S1.pkl", 'rb'))

RF_pred  = regressor.predict(test_dataset['features'])     

plt.figure(100)
plt.plot(RF_pred[:, 1], RF_pred[:, 0], 'r.', label='Predicted')
plt.title('Test data')
plt.xlabel('Valence')
plt.ylabel('Arousal')

plt.show()
