# Emotion recognition using physiological signals

### Installation instructions
The code is tested using python version 3.7.9. The required packages are present in the 'environment.yml' located in the main folder and can be installed in an anaconda environment. 

- Create a conda env with the name and python version mentioned in the environment.yml file
``` conda env create -f environment.yml```

- Activate conda environment, replace the {env_name} with the environment you just created.
``` conda activate {env_name} ```

Note: The package uses pytorch which requires cuda installation. The program will use CPU when no 'GPU' is found

### Folder structure
- **data**: contains raw (raw data files of the HRI, dreamer and Wesad datasets, also contains the Robot approaching experiment data in Hug_exp folder).
            interim, processed and features folders have the processed data in the form of dictionaries stored in h5 format. Run ```h5ls -r {file}``` to check the structure of data in a particular .h5 file. 

- **docs**: have the images foder that consists of the results obtained from regression
    - **[docs/CNN_architecture.md](docs/CNN_architecture.md)**: details about the CNN architecture
    - **[docs/HRI_data.md](docs/HRI_data.md)**: details about processing HRI_dataset
    - **[docs/results_report.md](docs/results_report.md)**: results 

- **models**: Containes the trained models
    - **SSL_models**: Trained models '.pth' format for Self-Supervised Representation learning.
                - model1_0 is for training the CNN model with architecture 1 as described in docs/CNN_architecture.md
                - model2_1 is for training the CNN model with architecture 2 as described in docs/CNN_architecture.md
    Note: We use model2_1 for downstream emotion recognition task, **MultiModal_model_0** represents the model trained on pooled data from all 11 subjects and randomly selecting 80% of the data for training and remaining 20% for testing. **MultiModal_model_1** represents Inter-Subject performance where the model is trained on 9 subjects and tested on 2 new subjects.

    - **EMOTION_models**: Trained models for downstream emotion recognition. 

    - **Regression_models**: This folder consists of all the Random forest models trained. For example, models trained on multi-modal feature set for individual subjects, models trained on individual subjects for each physiological signal, etc.

- **results**

- **runs**: run ```tensorboard --logdir=runs``` from the emotion_estimation directory for logging results to tensorboard and check the convergence plots of trained Neural Network models when you open ```http://localhost:6006/``` in a browser.

- **src**: contains all the code for data processing, feature extraction and training and testing models. The main.py file contains the multiple skip_run blocks. By default all of them are marked as 'skip', change it to 'run' to run a particular block of code.

- **test**: contains some test codes for signal transformations part