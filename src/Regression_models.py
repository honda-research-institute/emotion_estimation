import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

def remove_zero_rows_from_data(features, labels):
    """Clean the dataset by completely removing a row that holds zero value for all its features 
    """
    ind = np.arange(features.shape[0])
    
    ind = ind[np.linalg.norm(features, axis=1) > 0]
    
    cleaned_feats  = features[ind, :]
    cleaned_labels = labels[ind, :]

    return cleaned_feats, cleaned_labels

def train_regression_model(modality, train_data, regressor, clean_features=True, model_save_path=[]):
    """Train the regressor on the inpu training dataset and provide the test results

    Args:
        modality (str): The modality to use for training the regressor. Choose from ('ECG', 'EMG', 'GSR', 'PPG', 'RSP')
        train_data (dictionary): train dataset 
        regressor (obj): sklearn regressor
        model_save_path (str): path to save the pickle model
        pic_save_path (str): path to save the picture
    """

    if clean_features:
        x_train, y_train = remove_zero_rows_from_data(train_data['features'], train_data['labels'])
    else:
        x_train = train_data['features']
        y_train = train_data['labels']

    # train the regressor
    regressor.fit(x_train, y_train)

    # predict data
    train_pred = regressor.predict(x_train)

    results = { 'Dataset': ['Train', 'Train', 'Train'],
                'Affect': ['Arousal', 'Valence', 'Total'],
                'MSE': [mean_squared_error(y_train[:, 0], train_pred[:, 0]),
                        mean_squared_error(y_train[:, 1], train_pred[:, 1]), 
                        mean_squared_error(y_train, train_pred)],

                'R^2': [r2_score(y_train[:, 0], train_pred[:, 0]),
                        r2_score(y_train[:, 1], train_pred[:, 1]), 
                        r2_score(y_train, train_pred)],

                'MAE': [mean_absolute_error(y_train[:, 0], train_pred[:, 0]),
                        mean_absolute_error(y_train[:, 1], train_pred[:, 1]),
                        mean_absolute_error(y_train, train_pred)],

                'STD': [np.std(np.abs(y_train[:, 0] - train_pred[:, 0])),
                        np.std(np.abs(y_train[:, 1] - train_pred[:, 1])),
                        np.std(np.abs(y_train - train_pred))]}
    
    # save the results in to a pandas dataframe
    df = pd.DataFrame(results)
    print('------------------- ', modality , ' -------------------')
    print(df)

    plt.figure()
    
    plt.plot(train_pred[:, 1], train_pred[:, 0], 'r.', label='Predicted')
    plt.plot(y_train[:, 1], y_train[:, 0], 'b.', label='True')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.legend(loc='lower left')
    plt.xlim([1.5, 6])
    plt.ylim([1.5, 6])
    
    if model_save_path:
        pickle.dump(regressor, open(model_save_path, 'wb'))

    return regressor

def train_test_regression_model(modality, train_data, test_data, regressor, clean_features=True, model_save_path=[], pic_save_path=[], results_save_path=[]):
    """Train the regressor on the inpu training dataset and provide the test results

    Args:
        modality (str): The modality to use for training the regressor. Choose from ('ECG', 'EMG', 'GSR', 'PPG', 'RSP')
        train_data (dictionary): train dataset
        test_data (dictionary): test dataset 
        regressor (obj): sklearn regressor
        model_save_path (str): path to save the pickle model
        pic_save_path (str): path to save the picture
    """

    if clean_features:
        x_train, y_train = remove_zero_rows_from_data(train_data['features'], train_data['labels'])
        x_test, y_test   = remove_zero_rows_from_data(test_data['features'], test_data['labels'])
    else:
        x_train = train_data['features']
        x_test  = test_data['features']
        y_train = train_data['labels']
        y_test  = test_data['labels']

    # train the regressor
    regressor.fit(x_train, y_train)

    # predict data
    train_pred = regressor.predict(x_train)
    test_pred  = regressor.predict(x_test)

    results = { 'Dataset': ['Train', 'Train', 'Test', 'Test', 'Test'],
                'Affect': ['Arousal', 'Valence', 'Arousal', 'Valence', 'Total'],
                'MSE': [mean_squared_error(y_train[:, 0], train_pred[:, 0]),
                        mean_squared_error(y_train[:, 1], train_pred[:, 1]), 
                        mean_squared_error(y_test[:, 0], test_pred[:, 0]), 
                        mean_squared_error(y_test[:, 1], test_pred[:, 1]),
                        mean_squared_error(y_test, test_pred)],

                'R^2': [r2_score(y_train[:, 0], train_pred[:, 0]),
                        r2_score(y_train[:, 1], train_pred[:, 1]), 
                        r2_score(y_test[:, 0], test_pred[:, 0]), 
                        r2_score(y_test[:, 1], test_pred[:, 1]),
                        r2_score(y_test, test_pred)],

                'MAE': [mean_absolute_error(y_train[:, 0], train_pred[:, 0]),
                        mean_absolute_error(y_train[:, 1], train_pred[:, 1]), 
                        mean_absolute_error(y_test[:, 0], test_pred[:, 0]), 
                        mean_absolute_error(y_test[:, 1], test_pred[:, 1]),
                        mean_absolute_error(y_test, test_pred)],

                'STD': [np.std(np.abs(y_train[:, 0] - train_pred[:, 0])),
                        np.std(np.abs(y_train[:, 1] - train_pred[:, 1])), 
                        np.std(np.abs(y_test[:, 0] - test_pred[:, 0])), 
                        np.std(np.abs(y_test[:, 1] - test_pred[:, 1])),
                        np.std(np.abs(y_test - test_pred))]}
    
    feat_imp = pd.DataFrame({'feature_importance': regressor.feature_importances_})

    # save the results in to a pandas dataframe
    df = pd.DataFrame(results)
    print('------------------- ', modality , ' -------------------')
    print(df)

    fig, ax = plt.subplots(1, 2)
    
    ax[0].plot(train_pred[:, 1], train_pred[:, 0], 'r.', label='Predicted')
    ax[0].plot(y_train[:, 1], y_train[:, 0], 'b.', label='True')
    ax[0].set_title('Training data')
    ax[0].set_xlabel('Valence')
    ax[0].set_ylabel('Arousal')
    ax[0].legend(loc='lower left')
    ax[0].set_xlim([1.5, 6])
    ax[0].set_ylim([1.5, 6])
    
    ax[1].plot(test_pred[:, 1], test_pred[:, 0], 'r.', label='Predicted')
    ax[1].plot(y_test[:, 1], y_test[:, 0], 'b.', label='True')
    ax[1].set_title('Test data')
    ax[1].set_xlabel('Valence')
    # ax[1].set_ylabel('Arousal')
    # ax[1].legend()
    ax[1].set_xlim([1.5, 6])
    ax[1].set_ylim([1.5, 6])

    fig.suptitle(modality + ' features')

    if model_save_path:
        pickle.dump(regressor, open(model_save_path, 'wb'))

    if pic_save_path:
        plt.savefig(pic_save_path)

    if results_save_path:
        df.to_csv(results_save_path)
        feat_imp.to_csv(str(results_save_path.split('.')[0] + '_feat_imp.csv'))

    # return the absolute arousal and valence error
    return np.abs(y_train[:, 0] - train_pred[:, 0]), np.abs(y_train[:, 1] - train_pred[:, 1]), np.abs(y_test[:, 0] - test_pred[:, 0]), np.abs(y_test[:, 1] - test_pred[:, 1])
    
def test_pretrained_regression_model(modality, train_data, test_data, regressor, clean_features=True, pic_save_path=[], results_save_path=[]):
    """Train the regressor on the inpu training dataset and provide the test results

    Args:
        modality (str): The modality to use for training the regressor. Choose from ('ECG', 'EMG', 'GSR', 'PPG', 'RSP')
        train_data (dictionary): train dataset
        test_data (dictionary): test dataset 
        regressor (obj): sklearn regressor
        pic_save_path (str): path to save the picture
    """

    if clean_features:
        x_train, y_train = remove_zero_rows_from_data(train_data['features'], train_data['labels'])
        x_test, y_test   = remove_zero_rows_from_data(test_data['features'], test_data['labels'])
    else:
        x_train = train_data['features']
        x_test  = test_data['features']
        y_train = train_data['labels']
        y_test  = test_data['labels']

    # predict data
    train_pred = regressor.predict(x_train)
    test_pred  = regressor.predict(x_test)


    results = { 'Dataset': ['Train', 'Train', 'Test', 'Test', 'Test'],
                'Affect': ['Arousal', 'Valence', 'Arousal', 'Valence', 'Total'],
                'MSE': [mean_squared_error(y_train[:, 0], train_pred[:, 0]),
                        mean_squared_error(y_train[:, 1], train_pred[:, 1]), 
                        mean_squared_error(y_test[:, 0], test_pred[:, 0]), 
                        mean_squared_error(y_test[:, 1], test_pred[:, 1]),
                        mean_squared_error(y_test, test_pred)],
                        
                'R^2': [r2_score(y_train[:, 0], train_pred[:, 0]),
                        r2_score(y_train[:, 1], train_pred[:, 1]), 
                        r2_score(y_test[:, 0], test_pred[:, 0]), 
                        r2_score(y_test[:, 1], test_pred[:, 1]),
                        r2_score(y_test, test_pred)],

                'MAE': [mean_absolute_error(y_train[:, 0], train_pred[:, 0]),
                        mean_absolute_error(y_train[:, 1], train_pred[:, 1]), 
                        mean_absolute_error(y_test[:, 0], test_pred[:, 0]), 
                        mean_absolute_error(y_test[:, 1], test_pred[:, 1]),
                        mean_absolute_error(y_test, test_pred)],

                'STD': [np.std(np.abs(y_train[:, 0] - train_pred[:, 0])),
                        np.std(np.abs(y_train[:, 1] - train_pred[:, 1])), 
                        np.std(np.abs(y_test[:, 0] - test_pred[:, 0])), 
                        np.std(np.abs(y_test[:, 1] - test_pred[:, 1])),
                        np.std(np.abs(y_test - test_pred))]} 

    feat_imp = pd.DataFrame({'feature_importance': regressor.feature_importances_})

    # save the results in to a pandas dataframe
    df = pd.DataFrame(results)
    print('------------------- ', modality , ' -------------------')
    print(df)

    fig, ax = plt.subplots(1, 2)
    # ax[0].plot(train_pred[:, 1], train_pred[:, 0], 'r.', label='Predicted')
    # ax[0].plot(y_train[:, 1], y_train[:, 0], 'b.', label='True')
    # ax[0].set_title('Training data')
    # ax[0].set_xlabel('Valence')
    # ax[0].set_ylabel('Arousal')
    # # ax[0].legend(loc='lower left')

    # ax[1].plot(test_pred[:, 1], test_pred[:, 0], 'r.', label='Predicted')
    # ax[1].plot(y_test[:, 1], y_test[:, 0], 'b.', label='True')
    # ax[1].set_title('Test data')
    # ax[1].set_xlabel('Valence')
    # ax[1].set_ylabel('Arousal')
    # ax[1].legend()

    fig.suptitle('RF')

    ind = np.arange(0, y_train.shape[0])
    err = np.abs(y_train[:, 0] - train_pred[:, 0]) + np.abs(y_train[:, 1] - train_pred[:, 1])
    
    ind1 = ind[err <= 1]
    color1 = [0, 0, 1]

    ind2 = ind[(err > 1) & (err <= 2)]
    color2 = [0, 1, 0]
    
    ind3 = ind[err > 2]
    color3 = [1, 0, 0]

    ax[0].plot(y_train[ind1, 1], y_train[ind1, 0], '.', color=color1)
    ax[0].plot(y_train[ind2, 1], y_train[ind2, 0], '.', color=color2)
    ax[0].plot(y_train[ind3, 1], y_train[ind3, 0], '.', color=color3)

    ax[0].set_xlabel('Valence')
    ax[0].set_ylabel('Arousal')
    
    ind = np.arange(0, y_test.shape[0])
    err = np.abs(y_test[:, 0] - test_pred[:, 0]) + np.abs(y_test[:, 1] - test_pred[:, 1])
    
    ind1 = ind[err <= 1]
    color1 = [0, 0, 1]

    ind2 = ind[(err > 1) & (err < 2)]
    color2 = [0, 1, 0]
    
    ind3 = ind[err > 2]
    color3 = [1, 0, 0]

    ax[1].plot(y_test[ind1, 1], y_test[ind1, 0], '.', color=color1)
    ax[1].plot(y_test[ind2, 1], y_test[ind2, 0], '.', color=color2)
    ax[1].plot(y_test[ind3, 1], y_test[ind3, 0], '.', color=color3)

    ax[1].set_xlabel('Valence')
        
    if pic_save_path:
        plt.savefig(pic_save_path)

    if results_save_path:
        df.to_csv(results_save_path)
        feat_imp.to_csv(str(results_save_path.split('.')[0] + '_feat_imp.csv'))
    
    return np.abs(y_train[:, 0] - train_pred[:, 0]), np.abs(y_train[:, 1] - train_pred[:, 1]), np.abs(y_test[:, 0] - test_pred[:, 0]), np.abs(y_test[:, 1] - test_pred[:, 1])