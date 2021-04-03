# Feature extraction
Two types of feature extraction methods are used 1) Handcrafted features (Time and Frequency domain features), and 2) [Self supervised representations using CNN](CNN_architecture.md)

## Handcrafted features 

### ECG and PPG 
Heart Rate Variability (HRV) time and nonlinear features extracted from [Neurokit2](https://neurokit2.readthedocs.io/en/latest/functions.html#)

### GSR:
Mean, STD, Min, Max of cleaned [GSR signal](https://neurokit2.readthedocs.io/en/latest/functions.html#), Phasic and Tonic components, Skin Conductance Response (SCR) onsets, sum of amplitude, sum of rise times, and auto correlation of signal with itself at different time lags. 


### EMG:
The EMG features are extracted using [pysiology package](https://pysiology.readthedocs.io/en/latest/electromiography.html) -> analyzeEMG function in electromyography.py.
Note: LOG in time domain features and FR in frequency domain features are not considered.

    | Time Domain features  | Frequency domain features |
    | --------------------- | ------------------------- |
    |   IEMG                |       MNF                 |
    |   MAV                 |       MDF                 |
    |   MAV1                |       PeakFrequency       |
    |   MAV2                |       MNP                 |
    |   SSI                 |       TTP                 |
    |   VAR                 |       SM1                 |
    |   TM3                 |       SM2                 |
    |   TM4                 |       SM3                 |
    |   TM5                 |       PSR                 |
    |   RMS                 |       VCF                 |
    |   WL                  |       SM2                 |
    |   AAC                 |       -                   |
    |   DASDV               |       -                   |
    |   AFB                 |       -                   |
    |   ZC                  |       -                   |
    |   MYOP                |       -                   |
    |   WAMP                |       -                   |
    |   SSC                 |       -                   |
    |   MAVSLPk             |       -                   |
    |   HIST                |       -                   |
    

## Self supervised representations of ECG
The idea for extracting self supervised representations from ECG signal is obtained from the paper: "[Self-supervised ECG Representation Learning for Emotion Recognition](https://ieeexplore.ieee.org/document/9161416)"

- The trained network are SelfSupervisedNet and SelfSupervisedNet2 in NN_models.py where CNN blocks are same in both the architectures but the fully connected layers are different. In SelfSupervisedNet, one set of fully connected network is trained for each transformation as described in the paper "[Self-supervised ECG Representation Learning for Emotion Recognition](https://ieeexplore.ieee.org/document/9161416)". In SelfSupervisedNet2, an alternative approach is employed to train a single fully connected network with 7 outputs. 

- Each of the networks have a common convolutional block that applies a series of learnt 1D convolution filters to the time-series ECG signal to extract feature vector of length 128. More details about the architecture can be found in the [CNN_architecture.md](CNN_architecture.md) file. These features perform better than the handcrafted features.

Currently, self-supervised representations are only obtained from ECG signal
TODO: Build representation learning networks for all the physiological signals
