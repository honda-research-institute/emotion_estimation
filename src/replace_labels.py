import sys
import os
import numpy as np
import deepdish as dd
import collections

# take data and labels from different files and save to a single file
label_src_file = 'data/interim/hri_transition_8_0.h5'
#data_src_file = 'data/features/hri_ECG_SSL_feats_transition_0_0.h5'
#save_file = 'data/features/hri_ECG_SSL_feats_transition_8_0.h5'
data_src_file = 'data/features/hri_feats_transition_0_0.h5'
save_file = 'data/features/hri_feats_transition_8_0.h5'

label_src = dd.io.load(label_src_file)
data_src = dd.io.load(data_src_file)

data_out = collections.defaultdict(dict)

# replace labels in data_src with labels in label_src
for sub in data_src.keys():
    print('sub:', sub)
    data_sub = collections.defaultdict(dict)
    for e in data_src[sub].keys():
        print('event:', e)
        data_e = collections.defaultdict(dict)
        for m in data_src[sub][e].keys():
            print('modality:', m)
            if m == 'labels':
                data_e[m] = label_src[sub][e][m]
            else:
                data_e[m] = data_src[sub][e][m]

        data_sub[e] = data_e

    data_out[sub] = data_sub

dd.io.save(save_file, data_out)

