import numpy as np
import math
import random
import cv2
import samplerate

# check this paper for the types of noise encountered in ECG signals
# "Signal Processing Techniques for Removing Noise from ECG Signals"

# Prefer the add_noise_with_SNR function
def add_noise(signal, noise_param):
    """add gaussian noise with zero mean and a noise_param standard deviation

    Parameters
    ----------
    signal : numpy array
        input signal (N x 1)
    noise_param : float
        amount of noise to be added to the signal

    Returns
    -------
    numpy array
        signal + noise (N x 1)
    """
    noise = np.random.normal(0, noise_param, max(signal.shape))
    noised_signal = signal+noise
    return noised_signal.reshape(-1, 1)

def add_noise_with_SNR(signal, noise_param):
    """Additive White Gaussian Noise based on a selected SNR ratio as given in :
    created using: https://stackoverflow.com/a/53688043/10700812 

    Parameters
    ----------
    signal : numpy array
        input signal (N x 1)
    noise_param : float
        Signal to Noise Ration in dB 

    Returns
    -------
    numpy array
        signal + noise (N x 1)
    """
    
    target_snr_db = noise_param #20
    x_watts = signal ** 2                       # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)   # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), max(signal.shape)).reshape(-1, 1)  # Generate an sample of white noise
    noised_signal = signal + noise_volts        # noise added signal

    return noised_signal.reshape(-1, 1)


def scaled(signal, factor):
    """Scale the signal 

    Parameters
    ----------
    signal : numpy array 
        input signal (N x 1)
    factor : float
        scale the input signal by this factor

    Returns
    -------
    numpy array
        scaled signal (N x 1)
    """
    scaled_signal = signal * factor
    return scaled_signal.reshape(-1, 1)

def negate(signal):
    """multiply the input signal by -1

    Parameters
    ----------
    signal : numpy array 
        input signal (N x 1)

    Returns
    -------
    numpy array (N x 1)
    """
    negated_signal = signal * (-1)
    return negated_signal.reshape(-1, 1)

    
def hor_filp(signal):
    """flip the signal from front to back

    Parameters
    ----------
    signal : numpy array 
        input signal (N x 1)

    Returns
    -------
    numpy array (N x 1)
    """
    hor_flipped = np.flip(signal)
    return hor_flipped.reshape(-1, 1)


def split_signal(signal, pieces):
    """split the signal into bin (pieces) 

    Parameters
    ----------
    signal : numpy array
        input signal (Nx1)
    pieces : int
        number of pieces to split the signal
    """
    
    split_length = max(signal.shape) // pieces
    
    start_ind    = np.arange(0, max(signal.shape), split_length).tolist()
    end_ind      = np.arange(split_length, max(signal.shape), split_length).tolist()
    end_ind.append(max(signal.shape)) # append the last missing index
    
    # split the signal into blocks
    signal_blocks = [signal[start_ind[i]:end_ind[i]] for i in range(0, len(start_ind))]
    
    return signal_blocks
    
def permute(signal, pieces):
    """randomly shuffle the signal bins and concatenate them

    Parameters
    ----------
    signal : numpy array
        input signal (Nx1)
    pieces : int
        number of pieces to split the signal
    """
    signal_blocks = split_signal(signal, pieces)
    
    # randomly shuffle the signal blocks
    random.shuffle(signal_blocks)
    
    # concatenate the signal back
    shuffled_signal    = np.concatenate(signal_blocks, axis=0)
    
    return shuffled_signal.reshape(-1, 1)
    

def time_warp(signal, tw_pieces, stretch_factor, shrink_factor):
    """[summary]
    resampling based on : Erik de Castro Lopo’s libsamplerate
    https://stackoverflow.com/questions/29085268/resample-a-numpy-array

    Parameters
    ----------
    signal : numpy array
        input signal (Nx1)
    tw_pieces : int
        number of pieces to split the signal
    stretch_factor : float
        factor by which the time axis of signal should be streched
    shrink_factor : float
        factor by which the time axis of signal should be shrinked

    Returns
    -------
    numpy array (Nx1)
        warped signal created by splitting the signal into bins 
        then randomly streching and shrinking different bins and
        then reform the signal 
    """
    
    signal_blocks = split_signal(signal, tw_pieces)
    
    sequence = np.arange(0, len(signal_blocks)).tolist()
    random.shuffle(sequence)
    
    for i in sequence:
        if i < math.ceil(len(sequence)/2):
            factor = stretch_factor
        else:
            factor = shrink_factor
        
        signal_blocks[i] = samplerate.resample(signal_blocks[i], factor, 'sinc_best') # options: 'sinc_best' or 'sinc_fastest'
        
    warped_signal = np.concatenate(signal_blocks, axis=0)
    
    if len(warped_signal) > len(signal):
        warped_signal = warped_signal[:len(signal)]
    elif len(warped_signal) < len(signal):
        pad_length = int(len(signal) - len(warped_signal))
        warped_signal = np.concatenate((warped_signal, np.zeros((pad_length, 1))), axis=0)
    
    return warped_signal.reshape(-1, 1)


def apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_pieces, stretch_factor, shrink_factor):
    """ Apply all the signal transformations and return a list of the transformed signals along with the original signal"""

    if len(signal.shape) < 2:
        signal = signal.reshape(-1, 1) 

    x_noise     = add_noise_with_SNR(signal, noise_param)

    x_scaled    = scaled(signal, scale_param)

    x_negated   = negate(signal)

    x_fliped    = hor_filp(signal)

    x_perm      = permute(signal, permu_param)

    x_tw        = time_warp(signal, tw_pieces, stretch_factor, shrink_factor)

    x_signal    = signal 

    signal_transforms = [x_signal, x_noise, x_scaled, x_negated, x_fliped, x_perm, x_tw]
    signal_labels     = [0       , 1      , 2       , 3        , 4       , 5     , 6]

    signal_transforms = np.concatenate(signal_transforms, axis=1).T
    signal_labels     = np.array(signal_labels).reshape(-1, 1)

    return signal_transforms, signal_labels

        
        
        
    
    
    
    