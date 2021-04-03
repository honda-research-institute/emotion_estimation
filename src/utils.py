import os
import sys
import numpy as np
import signal_transformation as sgtf
from contextlib import contextmanager
from scipy import signal 

def makedirs(path):
    """ 
    create directory on the "path name" """
    
    if not os.path.exists(path):
        os.makedirs(path)

def import_filenames(directory_path):
    """ 
    import all file names of a directory """
    filename_list = []
    dir_list      = []
    for _, dirs, files in os.walk(directory_path, topdown=False):
        filename_list   = files     
        dir_list        = dirs
    return filename_list, dir_list

def pool_transformations(signal, noise_param, scale_param, n_bins, stretch_factor, shrink_factor):
    """pool the transformed signals along with the original signal

    Args:
        signal (numpy array): a stream of time series signal

    Returns:
        numpy array: an array of original signal concatenated with transformed signals
    """
    signal_transforms = []
    
    x_0 = signal
    x_1 = sgtf.add_noise_with_SNR(signal, noise_param)
    x_2 = sgtf.scaled(signal, scale_param)
    x_3 = sgtf.negate(signal)
    x_4 = sgtf.hor_filp(signal)
    x_5 = sgtf.permute(signal, n_bins)
    x_6 = sgtf.time_warp(signal, n_bins, stretch_factor, shrink_factor)

    signal_transforms.append(x_0, x_1, x_2, x_3, x_4, x_5, x_6)
 
    return np.concatenate(signal_transforms, axis=0)

def butter_highpass(cutoff, fs, order=5):
    """high pass filter the input data

    Args:
        cutoff (float): cut-off frequency
        fs (float): sampling frequency
        order (int, optional): order of the filter. Defaults to 5.

    Returns:
        coefficients of the butter filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """high pass filter the input data

    Args:
        data (ndarray): data to be filtered
        cutoff (float): cut-off frequency
        fs (float): sampling frequency
        order (int, optional): order of the filter. Defaults to 5.

    Returns:
        ndarray: filtered data
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


class SkipWith(Exception):
    pass

@contextmanager
def skip_run(flag, f):
    """To skip a block of code.
    Parameters
    ----------
    flag : str
        skip or run.
    Returns
    -------
    None
    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>8}  {:>2}  {:>12}'.format(
                'Skip', ':', f))
            raise SkipWith()
        else:
            p.print_run('{:>8}  {:>3}  {:>12}'.format('Run',
                                                       ':', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def balance_labels(labels):
    """ Return balanced indices based on Mike's image category"""
    ind = np.arange(labels.shape[0])

    # get the indices of neutral images and balance it with high valence and low valence images
    neutral_ind  = ind[(labels[:,0]<=2.5) & (labels[:,1]>=2.5) & (labels[:,1]<=4.5)]
    # positive_ind = ind[(labels[:,0]>2.5) & (labels[:,1]<4)]
    # negative_ind = ind[(labels[:,0]>2.5) & (labels[:,1]>4)]
    
    positive_ind = ind[(labels[:,0]>4) & (labels[:,1]<3.5)]
    negative_ind = ind[(labels[:,0]>4) & (labels[:,1]>4.5)]
    
    # randomly shuffle the indices
    np.random.shuffle(neutral_ind)
    np.random.shuffle(positive_ind)
    np.random.shuffle(negative_ind)

    # find the category with the least number of indices
    min_feat_len = np.min([len(neutral_ind), len(positive_ind), len(negative_ind)])
    
    # balance the data
    neutral_ind  = neutral_ind[:min_feat_len] 
    positive_ind = positive_ind[:min_feat_len]
    negative_ind = negative_ind[:min_feat_len]

    bal_ind = np.concatenate((neutral_ind, positive_ind, negative_ind), axis=0)

    return bal_ind