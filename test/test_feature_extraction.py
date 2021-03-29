import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import neurokit2 as nk
import deepdish as dd

sys.path.append(str(Path(__file__).parents[1] / 'src/'))
import signal_transformation as sgtf


# T = 10
# t = np.linspace(0, T, 2560)
# x = np.load(Path(__file__).parents[1] / 'data/sample_ecg.npy')
# x = x[0, :len(t)].reshape(-1,)

# data = nk.epochs_create(x, sampling_rate=256)
# temp = nk.ecg_analyze(data, sampling_rate=256)

# temp = nk.ecg_intervalrelated(data, sampling_rate=256)
# print(temp)

# plt.plot(t, x, 'r')
# plt.plot(t[8*256:9*256], data['9']['Signal'], 'b')
# plt.show()

data = dd.io.load(Path(__file__).parents[1] /'data/interim/hri.h5')

T = 10
t = np.linspace(0, T, 250)

gsr_sample = data['S1']['event1']['GSR'][2, :, :].reshape(-1, )

processed_eda, info = nk.eda_process(gsr_sample, sampling_rate=25)

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

eda_corr = nk.eda_autocor(processed_eda['EDA_Clean'], sampling_rate=25)

# plt.figure()
# for subject in data:
#     for event in ['event1', 'event2']:
#         eda = data[subject][event]['GSR']
#         for i, dat in enumerate(eda):
#             plt.plot(nk.eda_clean(dat.reshape(-1, ), sampling_rate=25))
#             plt.pause(0.01)

# plt.figure()
# plt.plot(gsr_sample.T, 'r')
# plt.figure()
# plt.plot(dic[0]['EDA_Raw'], 'b')
# plt.plot(dic[0]['EDA_Clean'])
plt.show()
