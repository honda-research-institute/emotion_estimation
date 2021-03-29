## Results of EMOTION recognition

#### Components of the recognition network
- CNN block for **representation learning** 
- Fully connected block + pretrained CNN block for **Downstream task**


#### Dataset
 - **DREAMER, WESAD, HRI** for training CNN block (Self-supervised learning)

 - **HRI** for downstream **EMOTION** recognition

 #### Accuracy reported
 - Semi-supervised Representation learning for ECG-SSL features
    1) SSL1: trained 7 independent fully connected networks after CNN for binary classification of 7 transformations
        Average Testing accuracy: **99.56 %**
    2) SSL2: trained single fully connected with 7 outputs (transformation categories)
        Average Testing accuracy:  **99.10 %**

- Downstream (EMOTION Recognition)
    1) EcgNet : A fully connected Neural Network architecture to predict Valence and Arousal values from ECG-SSL features
    2) EmotionNet : A fully connected Neural Network architeture to predict Valence and Arousal values from ECG-SSL, EMG, GSR, PPG features
    3) Random Forest Regressor : random forest regressors are trained for individual signals and also multimodal features


