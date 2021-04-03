import os 
import numpy as np
import pandas as pd
import csv 
import yaml
import utils 
import pickle
import collections
from scipy import io, stats
import mne 
import sys

from pathlib import Path
from tqdm import tqdm 
import deepdish as dd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import signal 
from sklearn.model_selection import train_test_split

# load the parameters from yaml file
# config_path = Path(__file__).parents[1] / 'config.yml'
# config = yaml.load(open(config_path), Loader=yaml.SafeLoader)



def process_ecg_mne(data, data_name, ch_names, ch_types, percent_overlap, sfreq, config):
    scaler = StandardScaler()

    # create mne info 
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info['description'] = data_name + 'dataset'

    # create mne raw object
    raw = mne.io.RawArray(data, info, verbose='error')
    
    # resample and high pass filter the data
    resamp_data   = raw.copy().resample(sfreq=config['freq'], verbose='error')
    # resamp_data   = resamp_data.copy().filter(l_freq=config['filter_band'][0], h_freq=config['filter_band'][1], method='iir', picks='misc', verbose='error')
    
    # high pass filter the data
    filtered_data = utils.butter_highpass_filter(resamp_data.get_data(), cutoff=0.5, fs=config['freq'], order=5)

    # normalize the data using z-score
    transformed_data = scaler.fit_transform(filtered_data.T).T
    # transformed_data1 = stats.zscore(resamp_data.get_data(), axis=1)

    # create mne raw object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=config['freq'])
    transformed_mne = mne.io.RawArray(transformed_data, info, verbose='error')

    # segment the data into fixed length windows
    if percent_overlap == 0:
        epochs = mne.make_fixed_length_epochs(transformed_mne, duration=config['window_size'], proj=False, verbose='error')
    else:
        events = mne.make_fixed_length_events(transformed_mne,
                                            duration=config['window_size'],
                                            overlap=percent_overlap*config['window_size'])
        epochs = mne.Epochs(transformed_mne,
                            events,
                            tmin=0,
                            tmax=config['window_size'],
                            baseline=None,
                            verbose='error')

    return epochs

def process_labels_mne(data, data_name, ch_names, ch_types, sfreq, config):
    # create mne info 
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    info['description'] = data_name + 'labels'

    # create mne raw object
    raw = mne.io.RawArray(data, info, verbose='error')
    
    # resample and high pass filter the data
    resamp_data   = raw.copy().resample(sfreq=config['freq'], verbose='error')

    # segment the data into fixed length windows
    epochs = mne.make_fixed_length_epochs(resamp_data, duration=config['window_size'], proj=False, verbose='error')

    return epochs

def read_raw_wesad_dataset(load_path, save_path, config, save=True):
    """read the WESAD data from respective .pkl files in the subject folders and save it into hdf5 format

    Args:
        load_path ([string]): path of the raw WESAD dataset
        save_path ([string]): path to store the WESAD dataset
        save ([bool]): boolean to save the data dictionary

    Returns:
        Data [dictionary]: Dictionary of chest (RespiBAN) and wrist (Empatica E4) sensor data. 
    """
    Data = collections.defaultdict(dict)
    
    if os.path.exists(load_path): 
        _, sub_list = utils.import_filenames(load_path)
        for sub in sub_list:
            data_dic = collections.defaultdict(dict)
            filepath = load_path + '//' + sub + '//' + sub + '.pkl' 
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                ecg  = data['signal']['chest']['ECG'].T
                # resp = data['signal']['chest']['Resp'].T
                labels = data['label'].reshape(1, -1)

                # temp = np.array((ecg, resp))
                data_dic['ECG'] = process_ecg_mne(ecg, 'wesad', ['ECG'], ['misc'], 0.5, config['wesad']['sfreq'], config).get_data()[:, :, :-1]
                data_dic['labels'] = process_labels_mne(labels, 'wesad', ['label'], ['misc'], config['wesad']['sfreq'], config).get_data()
                
            Data[sub] = data_dic

    else:
        print("Please check the path")
    
    if save:
        dd.io.save(save_path, Data)

    return Data
    
def read_raw_dreamer_dataset(load_path, save_path, config, save=True):
    """read the DREAMER data from .mat file and save it into hdf5 format

    Args:
        load_path ([string]): path of the raw DREAMER dataset
        save_path ([string]): path to store the DREAMER dataset
        save ([bool]): boolean to save the data dictionary

    Returns:
        Data [dictionary]: dictionary loaded from the 
    """
    Data = collections.defaultdict(dict)
    
    if os.path.exists(load_path): 
        data = io.loadmat(load_path, simplify_cells=True)
     
        # extract the data of all 23 subjects
        for i in range(len(data['DREAMER']['Data'])):
            data_list  = []
            label_list = []
            data_dic = collections.defaultdict(dict)
                    
            # retrieve the ECG signals of all 18 videos for each subject
            for j in range(len(data['DREAMER']['Data'][i]['ECG']['stimuli'])):
                
                ecg    = data['DREAMER']['Data'][i]['ECG']['stimuli'][j][:, 0].reshape(1, -1)
                
                # considering only one channel of ECG
                ecg = process_ecg_mne(ecg, 'dreamer', ['ECG1'], ['misc'], 0.5, config['dreamer']['sfreq'], config).get_data()[:, :, :-1]
                
                val = data['DREAMER']['Data'][i]['ScoreValence'][j] * np.ones((data['DREAMER']['Data'][i]['ECG']['stimuli'][j].shape[0], 1))
                ars = data['DREAMER']['Data'][i]['ScoreArousal'][j] * np.ones((data['DREAMER']['Data'][i]['ECG']['stimuli'][j].shape[0], 1))
                # dom = data['DREAMER']['Data'][i]['ScoreDominance'][j] * np.ones((data['DREAMER']['Data'][i]['ECG']['stimuli'][j].shape[0], 1))
                labels = np.concatenate((val, ars), axis=1).T
                labels = process_labels_mne(labels, 'dreamer', ['VALENCE', 'AROUSAL'], ['misc', 'misc'], config['dreamer']['sfreq'], config).get_data()
                
                data_list.append(ecg)
                label_list.append(labels)
            
            data_dic['ECG']   = np.concatenate(data_list, axis=0)
            # data_dic['label'] = np.concatenate(label_list, axis=0)

            Data['S'+ str(i+1)] = data_dic

    else:
        print("Please check the path")
    
    if save:
        dd.io.save(save_path, Data)

    return Data

def parse_hri_data_using_events(modality, filepath, col_names, skiprows, usecols, event_time, percent_overlap, arousal, valence, config, standardize=False, pick_sample=None):
    epochs, labels = [], []
    df = pd.read_csv(filepath,
                    delimiter=',',
                    names=col_names,
                    skiprows=skiprows,
                    usecols=usecols,
                    dtype=np.float32)
    df['time'] -= df['time'].iloc[0]

    # high pass filter the ecg data
    if modality == 'ecg':
        df[modality] = utils.butter_highpass_filter(df[modality].to_numpy(), cutoff=0.5, fs=config['hri']['sfreq'][modality], order=5)

    if standardize:
        scaler = StandardScaler()
        # standardize all modalities
        if modality == 'emg':
            for key in df.keys():
                if key != 'time':
                    df[key] = scaler.fit_transform(df[key].to_numpy().reshape(-1, 1))
        # elif (modality == 'ecg') or (modality == 'gsr'):
        else:
            df[modality] = scaler.fit_transform(df[modality].to_numpy().reshape(-1, 1))

    event_len = config['hri']['event_window'] * config['hri']['sfreq'][modality]
    epoch_len = config['window_size'] * config['hri']['sfreq'][modality]

    # split the data based on the events mentioned in pics.csv
    for i in range(len(event_time)):
        if (i == len(event_time)-1): 
            start_time = event_time[i]
            end_time   = df['time'].iloc[-1]
        else:
            start_time = event_time[i]
            end_time   = event_time[i+1]
            
        event_data = df.loc[(df['time'] >= start_time) & (df['time'] < end_time), col_names[1:]].to_numpy().reshape(len(col_names)-1, -1)

        # make sure that there is data for 15 sec otherwise pad the starting and ending with zeros
        if np.max(event_data.shape) < event_len:
            diff = event_len - np.max(event_data.shape)
            if (diff % 2) == 0:
                event_data = np.pad(event_data, ((0, 0), (int(diff/2), int(diff/2))), 'constant', constant_values=(0, 0))
            else:
                event_data = np.pad(event_data, ((0, 0), (int(diff // 2), int(diff // 2)+1)), 'constant', constant_values=(0, 0))
        
        # sfreq gives 1 sec window, (1-overlap_frac) * sfreq is less than 1 sec
        sliding_window = int((1 - percent_overlap) * config['hri']['sfreq'][modality])

        if pick_sample == None:
            for eve in range(0, event_len+1, sliding_window):
                if (event_len - eve) >= epoch_len:
                    epochs.append(event_data[:, eve:eve+epoch_len].reshape(1, len(col_names)-1, -1))
                    labels.append(np.array([arousal[i], valence[i]]).reshape(1, -1))

        elif pick_sample.lower() == 'first':
            for eve in range(0, 2*sliding_window+1, sliding_window):
                epochs.append(event_data[:, eve:eve+epoch_len].reshape(1, len(col_names)-1, -1))
                labels.append(np.array([arousal[i], valence[i]]).reshape(1, -1))

        elif pick_sample.lower() == 'last':
            for eve in range(event_len, event_len - (2*sliding_window+1), -sliding_window):
                epochs.append(event_data[:, eve-epoch_len:eve].reshape(1, len(col_names)-1, -1))
                labels.append(np.array([arousal[i], valence[i]]).reshape(1, -1))


    return epochs, labels

def parse_hri_data(modality, filepath, col_names, skiprows, usecols, arousal, valence, percent_overlap, config, standardize=False, input_scaler=None, event_time=[]):
    epochs, labels = [], []
    df = pd.read_csv(filepath,
                    delimiter=',',
                    names=col_names,
                    skiprows=skiprows,
                    usecols=usecols,
                    dtype=np.float32)
                    
    df['time'] -= df['time'].iloc[0]

    # high pass filter the ecg data
    if modality == 'ecg':
        df[modality] = utils.butter_highpass_filter(df[modality].to_numpy(), cutoff=0.5, fs=config['hri']['sfreq'][modality], order=5)

    scaler = input_scaler
    if standardize:
        if not input_scaler:
            scaler = StandardScaler()
        # standardize all modalities
        if modality == 'emg':
            for key in df.keys():
                if key != 'time':
                    df[key] = scaler.fit_transform(df[key].to_numpy().reshape(-1, 1))
        else:
            df[modality] = scaler.fit_transform(df[modality].to_numpy().reshape(-1, 1))

    epoch_len = config['window_size'] * config['hri']['sfreq'][modality]
    # sfreq gives 1 sec window, (1-overlap_frac) * sfreq is less than 1 sec
    sliding_window = int((1 - percent_overlap) * config['hri']['sfreq'][modality])

    if len(event_time) > 1:
        event_len = config['hri']['event_window'] * config['hri']['sfreq'][modality]

        # split the data based on the events mentioned in pics.csv
        for i in range(len(event_time)):
            if (i == len(event_time)-1): 
                start_time = event_time[i]
                end_time   = df['time'].iloc[-1]
            else:
                start_time = event_time[i]
                end_time   = event_time[i+1]

            event_data = df.loc[(df['time'] >= start_time) & (df['time'] < end_time), col_names[1:]].to_numpy().reshape(len(col_names)-1, -1)

            # make sure that there is data for 15 sec otherwise pad the starting and ending with zeros
            if np.max(event_data.shape) < event_len:
                diff = event_len - np.max(event_data.shape)
                if (diff % 2) == 0:
                    event_data = np.pad(event_data, ((0, 0), (int(diff/2), int(diff/2))), 'constant', constant_values=(0, 0))
                else:
                    event_data = np.pad(event_data, ((0, 0), (int(diff // 2), int(diff // 2)+1)), 'constant', constant_values=(0, 0))
        
            for eve in range(0, event_len+1, sliding_window):
                if (event_len - eve) >= epoch_len:
                    epochs.append(event_data[:, eve:eve+epoch_len].reshape(1, len(col_names)-1, -1))
                    labels.append(np.array([arousal[i], valence[i]]).reshape(1, -1))
    else:
        event_len = df['time'].shape[0] 
        start_time = df['time'].iloc[0]
        end_time   = df['time'].iloc[-1]

        event_data = df.loc[(df['time'] >= start_time) & (df['time'] < end_time), col_names[1:]].to_numpy().reshape(len(col_names)-1, -1)
        
        for eve in range(0, event_len+1, sliding_window):
            if (event_len - eve) > epoch_len:
                epochs.append(event_data[:, eve:eve+epoch_len].reshape(1, len(col_names)-1, -1))   

    return epochs, labels, scaler

# read the data from folder and create a dictionary for all the subjects
def read_raw_hri_dataset(load_path, save_path, percent_overlap, config, save=True, standardize=False, scenarios=[1, 2], pick_sample=None):
    """Extract the ECG, EMG, GSR, PPG, RSP data from the csv files of the HRI dataset 

    Args:
        load_path (string): path to the dataset
        save_path (string): path to save the data
        percent_overlap (float): overlap percentage of 1 sec window      
        config (dictionary): imported configuration from the config.yml file 
        save (bool, optional): flag to save the extracted data. Defaults to True.
        standardize (bool, optional): standardize (z-score) the individual data. Defaults to False.
        scenarios (list, optional): provide [1, 2] when trying to read the emotion data from each scenario, otherwise, provide [''] (This is to process the human-hugrobot interaction data)
        pick_sample (str): options:['first', 'last', None]
                        'first' : first 3 samples from the 15 s event
                        'last'  : last 3 samples from the 15 s event
                         None   : Use the complete 15 s event window to extract epochs

    Returns:
        dictionary: dictionary of imported data from the files
    """
    Data = collections.defaultdict(dict)

    for subject, dir in enumerate(os.listdir(Path(__file__).parents[1] / load_path)):
        data = collections.defaultdict(dict)
        
        # there are two scenarios
        for scenario_type in scenarios:
            event_dic = collections.defaultdict(dict)  
            event_time, valence, arousal = [], [], []

            for file in os.listdir(os.path.join(Path(__file__).parents[1], load_path, dir)):
                scenario = file.split(str(scenario_type) + '_')
                if scenario[-1].lower() == 'pics.csv':
                    file_path = os.path.join(Path(__file__).parents[1], load_path, dir, file)
                    dataframe = pd.read_csv(file_path,
                                            delimiter=',',
                                            names=['time', 'arousal', 'valence'],
                                            skiprows=1,
                                            usecols=[0,1,3])
                    arousal = dataframe['arousal'].to_numpy()
                    valence = dataframe['valence'].to_numpy()
                    event_time = dataframe['time'].to_numpy()

            for file in os.listdir(os.path.join(Path(__file__).parents[1], load_path, dir)):
                scenario = file.split(str(scenario_type) + '_')
                filepath = os.path.join(Path(__file__).parents[1], load_path, dir, file)

                if scenario[-1].lower() in config['hri']['file_names']:
                    modality = scenario[-1].lower().split('.')[0]
                    
                    if modality == 'ecg':
                        epochs, labels = parse_hri_data_using_events(modality, filepath, ['time', modality], 3, [0,1], event_time, percent_overlap, arousal, valence, config, standardize=standardize, pick_sample=pick_sample)
                    elif modality == 'emg':
                        epochs, labels = parse_hri_data_using_events(modality, filepath, ['time', 'emg1', 'emg2', 'emg3'], 3, [0,1,3,5], event_time, percent_overlap, arousal, valence, config, standardize=standardize, pick_sample=pick_sample)
                    elif modality == 'gsr':
                        epochs, labels = parse_hri_data_using_events(modality, filepath, ['time', modality], 2, [0,1], event_time, percent_overlap, arousal, valence, config, standardize=standardize, pick_sample=pick_sample)
                    elif modality == 'ppg':
                        epochs, labels = parse_hri_data_using_events(modality, filepath, ['time', modality], 2, [0,1], event_time, percent_overlap, arousal, valence, config, standardize=standardize, pick_sample=pick_sample)
                    elif modality == 'rsp':
                        epochs, labels = parse_hri_data_using_events(modality, filepath, ['time', modality], 2, [0,1], event_time, percent_overlap, arousal, valence, config, standardize=standardize, pick_sample=pick_sample)
                    
                    features = np.concatenate(epochs, axis=0)
                    labels   = np.concatenate(labels, axis=0)

                    if modality.lower() == 'ecg':
                        event_dic['labels'] = labels
                    
                    #FIXME: GSR files have some 'inf' in the files which are manually fixed
                    event_dic[modality.upper()] = features
                
            data['event'+ str(scenario_type)] = event_dic

        Data['S' + str(subject+1)] = data        

    if save:
        dd.io.save(save_path, Data)           

    return Data

# read the individual subject data from the provided path 
def read_individual_hri_dataset(load_path, save_path, percent_overlap, config, save=True, standardize=False, input_scaler=None, calib=False):
    """Extract the ECG, EMG, GSR, PPG, RSP data from the csv files of the HRI dataset 

    Args:
        load_path (string): path to the dataset
        save_path (string): path to save the data
        percent_overlap (float): overlap percentage of 1 sec window      
        config (dictionary): imported configuration from the config.yml file 
        save (bool, optional): flag to save the extracted data. Defaults to True.
        standardize (bool, optional): standardize (z-score) the individual data. Defaults to False.
        input_scaler(dictionary): dictionary of StandardScaler objects for each modality
        pick_sample (str): options:['first', 'last', None]
                        'first' : first 3 samples from the 15 s event
                        'last'  : last 3 samples from the 15 s event
                         None   : Use the complete 15 s event window to extract epochs

    Returns:
        dictionary: dictionary of imported data from the files
    """
    data = collections.defaultdict(dict)
    
    event_dic = collections.defaultdict(dict)  
    arousal, valence, event_time = [], [], []
    for file in os.listdir(os.path.join(Path(__file__).parents[1], load_path)):
        scenario = file.split('_')
        filepath = os.path.join(Path(__file__).parents[1], load_path, file)
        if calib:
            if scenario[-1].lower() == 'pics.csv':
                file_path = os.path.join(Path(__file__).parents[1], load_path, file)
                dataframe = pd.read_csv(file_path,
                                        delimiter=',',
                                        names=['time', 'arousal', 'valence'],
                                        skiprows=1,
                                        usecols=[0,1,3])
                arousal = dataframe['arousal'].to_numpy()
                valence = dataframe['valence'].to_numpy()
                event_time = dataframe['time'].to_numpy()
        else:
            event_time = []

    scaler_dict = {}      
    for file in os.listdir(os.path.join(Path(__file__).parents[1], load_path)):
        scenario = file.split('_')
        filepath = os.path.join(Path(__file__).parents[1], load_path, file)
        if scenario[-1].lower() in config['hri']['file_names']:
            modality = scenario[-1].lower().split('.')[0]
            if input_scaler:
                scaler=input_scaler[modality.upper()]
            else:
                scaler=None

            if modality == 'ecg':
                epochs, labels, scaler_dict[modality.upper()] = parse_hri_data(modality, filepath, ['time', modality], 3, [0,1], arousal, valence, percent_overlap, config, standardize=standardize, input_scaler=scaler, event_time=event_time)
            elif modality == 'emg':
                epochs, labels, scaler_dict[modality.upper()] = parse_hri_data(modality, filepath, ['time', 'emg1', 'emg2', 'emg3'], 3, [0,1,3,5], arousal, valence, percent_overlap, config, standardize=standardize, input_scaler=scaler, event_time=event_time)
            elif modality == 'gsr':
                epochs, labels, scaler_dict[modality.upper()] = parse_hri_data(modality, filepath, ['time', modality], 2, [0,1], arousal, valence, percent_overlap, config, standardize=standardize, input_scaler=scaler, event_time=event_time)
            elif modality == 'ppg':
                epochs, labels, scaler_dict[modality.upper()] = parse_hri_data(modality, filepath, ['time', modality], 2, [0,1], arousal, valence, percent_overlap, config, standardize=standardize, input_scaler=scaler, event_time=event_time)
            elif modality == 'rsp':
                epochs, labels, scaler_dict[modality.upper()] = parse_hri_data(modality, filepath, ['time', modality], 2, [0,1], arousal, valence, percent_overlap, config, standardize=standardize, input_scaler=scaler, event_time=event_time)
            
            features = np.concatenate(epochs, axis=0)

            if calib:
                labels   = np.concatenate(labels, axis=0)

            if modality.lower() == 'ecg':
                event_dic['labels'] = labels
            
            #FIXME: GSR files have some 'inf' in the files which are manually fixed
            event_dic[modality.upper()] = features
        
    data['event'] = event_dic
      
    if save:
        dd.io.save(save_path, data)           

    return data, scaler_dict


def split_data_train_test_valid(data, labels, test_size=0.2, shuffle=True, random_state=None):
    """Split the dataset into three parts training (default 80%), test (default 16%), and validation (default 4%)

    Args:
        data (ndarray): data (N samples x M features)
        labels (array): 1d array
        test_size (float, optional): % of data used for splitting data into test set, 
        same factor is then used to split the test set into test and validations sets. Defaults to 0.25.
        shuffle (bool, optional): if True, shuffles the data before splitting it. Defaults to True.
        random_state (int, optional): Random state used to perform similar split of the input data. Defaults to None.

    Returns:
        ndarray: train, test and validation sets split from the input data
    """
    train, test, y_train, y_test = train_test_split(data, labels, test_size=test_size, shuffle=shuffle, random_state=random_state)
    test, valid, y_test, y_valid = train_test_split(test, y_test, test_size=test_size, shuffle=shuffle, random_state=random_state)

    return train, test, valid, y_train, y_test, y_valid

def split_modalities_train_test_valid(data_dic, labels, test_size=0.2, shuffle=True, random_state=None):
    """Read each modality from the data dictionary and split it into three parts training (default 80%), test (default 16%), and validation (default 4%)
    with same order being followed for all the modalities
    Args:
        data_dic (ndarray): data (N samples x M features)
        labels (array): 1d array
        test_size (float, optional): % of data used for splitting data into test set, 
        same factor is then used to split the test set into test and validations sets. Defaults to 0.25.
        shuffle (bool, optional): if True, shuffles the data before splitting it. Defaults to True.
        random_state (int, optional): Random state used to perform similar split of the input data. Defaults to None.

    Returns:
        dictionary: train, test and validation sets  
    """
    ind = np.arange(labels.shape[0])

    _, _, train_ind, test_ind = train_test_split(ind, ind, test_size=test_size, shuffle=shuffle, random_state=random_state)
    _, _, test_ind, valid_ind = train_test_split(test_ind, test_ind, test_size=test_size, shuffle=shuffle, random_state=random_state)

    train    = {'ECG': data_dic['ECG'][train_ind, :, :],
                'EMG': data_dic['EMG'][train_ind, :, :],
                'GSR': data_dic['GSR'][train_ind, :, :],
                'PPG': data_dic['PPG'][train_ind, :, :],
                'RSP': data_dic['RSP'][train_ind, :, :],
                'labels': labels[train_ind, :]}
    
    test     = {'ECG': data_dic['ECG'][test_ind, :, :],
                'EMG': data_dic['EMG'][test_ind, :, :],
                'GSR': data_dic['GSR'][test_ind, :, :],
                'PPG': data_dic['PPG'][test_ind, :, :],
                'RSP': data_dic['RSP'][test_ind, :, :],
                'labels': labels[test_ind, :]}
    
    valid    = {'ECG': data_dic['ECG'][valid_ind, :, :],
                'EMG': data_dic['EMG'][valid_ind, :, :],
                'GSR': data_dic['GSR'][valid_ind, :, :],
                'PPG': data_dic['PPG'][valid_ind, :, :],
                'RSP': data_dic['RSP'][valid_ind, :, :],
                'labels': labels[valid_ind, :]}
    
    return train, test, valid





