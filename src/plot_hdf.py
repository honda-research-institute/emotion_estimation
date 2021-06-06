import sys
import os
import numpy as np
import deepdish as dd
import collections
import matplotlib.pyplot as plt

filename = 'data/interim/hri_transition_4_0.h5'
data = dd.io.load(filename)
subs = ['S4']
events = ['event1']
modalities = ['ECG']
window_idx = [0, 1]

count = 1
for sub in subs:
    for e in events:
        for mod in modalities:
            fig, ax = plt.subplots(len(window_idx))
            print(ax.shape)
            for i, w in enumerate(window_idx):
                if mod == 'labels':
                    d = data[sub][e][mod][w, :]
                else:
                    d = data[sub][e][mod][w, :, :]
                ax[i].set_xlabel('frame')
                ax[i].set_ylabel(mod)
                ax[i].plot(range(d.shape[1]), d.reshape(-1, d.shape[0]), '-')
                count += 1

            plt.show()
