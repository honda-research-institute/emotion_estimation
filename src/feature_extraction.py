import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import deepdish as dd
from pysiology import electromyography
from tqdm import tqdm 


def replace_inf_nan_with_zeros(df): 
    """Replace the 'inf', '-inf' and 'nan' values in a dataframe with zeros"""
    return df.replace(to_replace=[np.inf , -np.inf, np.nan], value=0)


def extract_ecg_features(ecg_signal, sampling_rate=200):
    """A total of 42 features are Extracted, 14-time-series and 28-nonlinear features from the ecg signal"""

    # cleaning step is avoided here as the ecg_signal is already high-pass filtered in the dataprocessing step
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    peaks, _    = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    
    if np.sum([peaks['ECG_R_Peaks'] == 1]) > 4:
        # print(np.sum([peaks['ECG_R_Peaks'] == 1]))
        time_feats  = nk.hrv_time(peaks, sampling_rate=sampling_rate)
        nonlinear_feats = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate)

        # The low frequency features are removed from this study as the ecg signal was
        # high-pass filtered 0.8 Hz in the dataprocessing step to remove baseline wander
        # freq_feats  = nk.hrv_frequency(peaks, sampling_rate=sampling_rate)

        time_feats      = replace_inf_nan_with_zeros(time_feats).to_numpy()
        # Last column - SampEn feature is removed from the features as it returns inf most of the time
        nonlinear_feats = replace_inf_nan_with_zeros(nonlinear_feats).to_numpy()[:, :-1] 

        ecg_features = np.concatenate([time_feats, nonlinear_feats], axis=1)
    
    else:
        ecg_features = np.zeros((1, 42))

    return ecg_features.reshape(1, 1, -1)


def extract_gsr_features(gsr_signal, sampling_rate=25):
    """12 statistical features are extracted from the GSR signal"""
    processed_eda, info = nk.eda_process(gsr_signal, sampling_rate=sampling_rate)

    processed_eda = replace_inf_nan_with_zeros(processed_eda)

    mean_eda = np.mean(processed_eda['EDA_Clean'])
    std_eda  = np.std(processed_eda['EDA_Clean'])
    min_eda  = np.min(processed_eda['EDA_Clean'])
    max_eda  = np.max(processed_eda['EDA_Clean'])

    mean_scr = np.mean(processed_eda['EDA_Phasic'])
    std_scr  = np.std(processed_eda['EDA_Phasic'])

    mean_scl = np.mean(processed_eda['EDA_Tonic'])
    std_scl  = np.std(processed_eda['EDA_Tonic'])

    scr_segments = len(info['SCR_Onsets'])

    sum_scr_amp  = np.sum(processed_eda['SCR_Amplitude'])
    sum_scr_rise = np.sum(processed_eda['SCR_RiseTime'])

    eda_corr = nk.eda_autocor(processed_eda['EDA_Clean'], sampling_rate=sampling_rate)

    features = np.array([mean_eda, std_eda, min_eda, 
                        max_eda, mean_scr, std_scr, 
                        mean_scl, std_scl, scr_segments, 
                        sum_scr_amp, sum_scr_rise, eda_corr]).reshape(1, -1)
    
    return features.reshape(1, 1, -1)



def extract_ppg_features(ppg_signal, sampling_rate=200):
    """"""
    # cleaning step is avoided here as the ecg_signal is already high-pass filtered in the dataprocessing step
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
    info        = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)

    if np.max(info['PPG_Peaks'].shape) > 3:
        ppg_processed,_ = nk.ppg_process(ppg_signal, sampling_rate=sampling_rate)
        time_feats  = nk.hrv_time(peaks=ppg_processed['PPG_Peaks'], sampling_rate=sampling_rate)
        nonlinear_feats = nk.hrv_nonlinear(peaks=ppg_processed['PPG_Peaks'], sampling_rate=sampling_rate)

        time_feats      = replace_inf_nan_with_zeros(time_feats).to_numpy()
        # Last column - SampEn feature is removed from the features as it returns inf most of the time
        nonlinear_feats = replace_inf_nan_with_zeros(nonlinear_feats).to_numpy()[:, :-1]

        ppg_features = np.concatenate([time_feats, nonlinear_feats], axis=1)
    
    else:
        ppg_features = np.zeros((1, 42))

    return ppg_features.reshape(1, 1, -1)


def extract_emg_features(emg_signal, sampling_rate=250):
    """ Extract 37 time domain features and 11 frequency domain features for each channel of the EMG
        output size - 48 (# of features) x 3 (channels) """
    emg_features = []
    for channel in range(emg_signal.shape[1]):
        processed_emg = electromyography.analyzeEMG(emg_signal[:, channel], samplerate=sampling_rate, lowpass=100, highpass=20)
        
        time_feats, freq_feats = [], []
        for key in processed_emg['TimeDomain'].keys():
            if key != 'LOG':
                if key == 'MAVSLPk':
                    time_feats + processed_emg['TimeDomain'][key]
                elif key == 'HIST':
                    for val in processed_emg['TimeDomain'][key]:
                        time_feats.append(processed_emg['TimeDomain'][key][val]['ZC'])
                        time_feats.append(processed_emg['TimeDomain'][key][val]['WAMP'])
                else:
                    time_feats.append(processed_emg['TimeDomain'][key])

        for key in processed_emg['FrequencyDomain'].keys():
            if key != 'FR': # see if FR feature should be inlcuded or not as sometimes it returns a Inf value, 
                freq_feats.append(processed_emg['FrequencyDomain'][key])

        total_feats = np.concatenate((time_feats, freq_feats), axis=0).reshape(-1, 1)
        total_feats = np.nan_to_num(total_feats, nan=0.0, posinf=0.0, neginf=0.0) # replace the nan and inf values with 0.0
        emg_features.append(total_feats)

    emg_features = (np.concatenate(emg_features, axis=1)).reshape(1, 3, -1)
    
    return emg_features


def extract_all_features(data_dic, config):
    hri_ecg, hri_emg, hri_gsr, hri_ppg, hri_rsp, hri_labels = [], [], [], [], [], []
    sample_freq = config['hri']['sfreq']

    ECG_epochs = data_dic['ECG']
    EMG_epochs = data_dic['EMG']
    GSR_epochs = data_dic['GSR']
    PPG_epochs = data_dic['PPG']
    # RSP_epochs = data_dic['RSP']

    [hri_ppg.append(extract_ppg_features(signal.reshape(-1, ), sampling_rate=sample_freq['ppg'])) for signal in tqdm(PPG_epochs)]
    [hri_ecg.append(extract_ecg_features(signal.reshape(-1, ), sampling_rate=sample_freq['ecg'])) for signal in tqdm(ECG_epochs)]
    [hri_emg.append(extract_emg_features(signal.reshape(-1, 3), sampling_rate=sample_freq['emg'])) for signal in tqdm(EMG_epochs)]
    [hri_gsr.append(extract_gsr_features(signal.reshape(-1, ), sampling_rate=sample_freq['gsr'])) for signal in tqdm(GSR_epochs)]
    # [hri_rsp.append(extract_rsp_features(signal.reshape(-1, ), sampling_rate=sample_freq['rsp'])) for signal in tqdm(RSP_epochs)]

    hri_labels.append(data_dic['labels'])

    hri_ecg     = np.concatenate(hri_ecg, axis=0)
    hri_emg     = np.concatenate(hri_emg, axis=0)
    hri_gsr     = np.concatenate(hri_gsr, axis=0)
    hri_ppg     = np.concatenate(hri_ppg, axis=0)
    # hri_rsp     = np.concatenate(hri_rsp, axis=0)

    hri_labels  = np.concatenate(hri_labels, axis=0)

    data = {'ECG': hri_ecg,
            'EMG': hri_emg,
            'GSR': hri_gsr,
            'PPG': hri_ppg,
            # 'RSP': hri_rsp,
            'labels': hri_labels}

    return data


#FIXME: This is not currently working
def extract_rsp_features(rsp_signal, sampling_rate=25):
    """ features are extracted from the RSP signal """
    
    rsp_cleaned = nk.rsp_clean(rsp_signal, sampling_rate=sampling_rate)

    # info = nk.rsp_findpeaks(rsp_cleaned, sampling_rate=sampling_rate)

    # print(info['RSP_Peaks'], info['RSP_Troughs'])

    # print(arr)

    positive = rsp_cleaned > 0
    negative = rsp_cleaned < 0

    risex = np.where(np.bitwise_and(negative[:-1], positive[1:]))[0]
    fallx = np.where(np.bitwise_and(positive[:-1], negative[1:]))[0]
    
    zc = np.concatenate((risex, fallx))
    zc.sort(kind="mergesort")

    print(zc)

    if (zc.shape[0] > 4):
        peak_signal,_ = nk.rsp_peaks(rsp_cleaned, sampling_rate=sampling_rate)

        info = nk.rsp_fixpeaks(peak_signal)
        # rsp_formatted = nk.signal_formatpeaks(info, desired_length=len(rsp_cleaned), peak_indices=info["RSP_Peaks"])

        # print(rsp_formatted)

        rsp_rate    = nk.rsp_rate(rsp_cleaned, sampling_rate=sampling_rate)
        rsp_feats   = nk.rsp_rrv(rsp_rate, peaks=info, sampling_rate=sampling_rate)

        print(rsp_feats.to_numpy().shape)
    # else:
    #     print(risex.shape[0], fallx.shape[0])
        
        # plt.cla()
        # plt.plot(rsp_cleaned)
        # plt.plot(risex, rsp_cleaned[risex], 'r*')
        # plt.plot(fallx, rsp_cleaned[fallx], 'b*')
        # plt.plot(np.arange(np.max(rsp_cleaned.shape)), 0*np.arange(np.max(rsp_cleaned.shape)))
        # plt.pause(.5)

    return rsp_cleaned.reshape(1, 1, -1)
