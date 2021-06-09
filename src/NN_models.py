import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os 
import sys
import numpy as np
import yaml
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import deepdish as dd
from tqdm import tqdm 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import utils
import signal_transformation as sgtf
from NN_datasets import EcgDataset

# load the configurations
config_path = Path(__file__).parents[1] / 'config.yml'
config = yaml.load(open(config_path), Loader=yaml.SafeLoader)

##########################################################################
# ------ Self Supervised Network Architectures ------ #
##########################################################################
# network module for self supervised representation learning using ECG
class SelfSupervisedNet(nn.Module):
    
    def __init__(self, device=torch.device("cpu"), config=[]):
        super(SelfSupervisedNet, self).__init__()
        
        # import configurations
        self.config = config

        self.num_flat_feat = config['torch']['SSL_flat_feat']
        self.device = device
        self.writer = SummaryWriter(log_dir=self.config['torch']['SSL_runs'] + "run1_" +
                                            str(len(os.listdir(self.config['torch']['SSL_runs'])))
                                            if os.path.exists(self.config['torch']['SSL_runs']) else 0)

        # convolutional layers
        self.conv_layer1 = self.conv_block(1, 32, 32, 1, 0)
        
        self.conv_layer2 = self.conv_block(32, 64, 16, 1, 0)

        self.conv_layer3 = self.conv_block(64, 128, 8, 1, 0)

        # fully connected layers for each task
        self.task0 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task1 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task2 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task3 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task4 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task5 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)

        self.task6 = self.fully_connect_block(self.num_flat_feat, 1, 0.5)
        
        # weight initialization of network
        self.weight_init()

        # objective function
        self.loss_criterion = nn.BCELoss(reduction='mean')

    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input 
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        conv = nn.Sequential(
            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob)
        )

        return conv

    def fully_connect_block(self, in_features, out_features, dropout_prob):
        
        dense = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=in_features, out_features=out_features),
            # nn.LeakyReLU(),
            # nn.Dropout(p=dropout_prob),

            nn.Sigmoid()
        )

        return dense

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):
        # self-supervised network
        x = self.conv_layer1(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer2(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer3(x)        
        x = F.max_pool1d(x, kernel_size=x.size()[-1]) # global max pooling
        # x = torch.amax(x, dim=-1) # alternative way of max pooling

        # flatten the output
        x = x.view(-1, self.num_flat_feat)

        # task-specific fully connected layers for each signal transformations
        x0 = self.task0(x)
        x1 = self.task1(x)
        x2 = self.task2(x)
        x3 = self.task3(x)
        x4 = self.task4(x)
        x5 = self.task5(x)
        x6 = self.task6(x)
        
        return [x0, x1, x2, x3, x4, x5, x6]
    
    def convert_labels(self, task_category, labels):
        """return 1's for the task corresponding to task_category"""

        true_labels = labels

        true_labels = torch.where(true_labels == task_category, 1, 0)

        return true_labels

    def total_loss(self, output, labels, task_weights):
        loss = 0
        for i, task in enumerate(output):
            true_labels = self.convert_labels(i, labels)

            loss += task_weights[i] * self.loss_criterion(task, true_labels.to(self.device, dtype=torch.float))
        
        return loss.to(self.device)
    
    def net_accuracy(self, output, labels):
        accuracy = np.zeros((self.config['n_transforms'])).tolist()
        total    = np.zeros((self.config['n_transforms'])).tolist()
        correct  = np.zeros((self.config['n_transforms'])).tolist()

        for i, task in enumerate(output):
            true_labels = self.convert_labels(i, labels).numpy()
            prediction  = torch.where(task>0.5, 1, 0).to('cpu').numpy()
            
            total[i] += len(true_labels) 
            correct[i] += (prediction == true_labels).sum().item()

        for i in range(0, self.config['n_transforms']):
            accuracy[i] = 100 * correct[i] / total[i]

        return accuracy

    def train_model(self, train_dataloader, validation_dataloader, optimizer, scheduler, batch_size, epochs, task_weights):
        model_save_path = self.config['torch']['SSL_models'] + 'model1_' + str(len(os.listdir(self.config['torch']['SSL_models'])))
        utils.makedirs(model_save_path)

        for epoch in range(epochs):
            # set the model to training mode
            self.train(mode=True)

            running_loss = 0
            counter = 0
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()

                output = self.forward(data['features'].to(self.device, dtype=torch.float))

                self.loss   = self.total_loss(output, data['labels'], task_weights)

                self.loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                optimizer.step()

                running_loss += self.loss.item()
                counter = i + 1
            
            # set the model to testing mode (Dropout and Batchnorms are disregarded)
            self.eval()

            train_accuracy = self.net_accuracy(output, data['labels'])
            
            print('Device: {}, Loss: {}, Accuracy: {} at epoch: {}'.format(self.device, running_loss/counter, train_accuracy, epoch+1))

            # Validation 
            valid_accuracy = []
            for valid_data in validation_dataloader:
                valid_output = self.forward(valid_data['features'].to(self.device)) # copy the validation data to cuda
                valid_data['features'].to(torch.device('cpu')) # remove the validation data from cuda
                # print(torch.cuda.memory_summary(device=self.device))
                temp_accuracy = self.net_accuracy(valid_output, valid_data['labels'])
                valid_accuracy.append(sum(temp_accuracy) / len(temp_accuracy))

            self.writer.add_scalar('Training Loss', running_loss/counter, epoch+1)
            self.writer.add_scalar('Training Accuracy', sum(train_accuracy)/len(train_accuracy), epoch+1)
            self.writer.add_scalar('Validation Accuracy', sum(valid_accuracy) / len(valid_accuracy), epoch+1)

            running_loss = 0
            if scheduler:
                scheduler.step()
           
            # save torch model
            if epoch % 100 == 99:
                # (str(len(os.listdir(self.config['torch']['SSL_models']))) if os.path.exists(self.config['torch']['SSL_models']) else 0) +
                model_path = model_save_path + '/net_' + str(epoch+1) + '.pth' 
                torch.save({ 'epoch': epoch+1,
                             'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': self.loss,
                            }, model_path)

        self.writer.close()
        print('Finished training')

    def validate_model(self, dataloader, optimizer, checkpoint, device=torch.device('cpu')):

        # load the model
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss  = checkpoint['loss']

        self.eval()
        
        mean_accuracy = []
        for data in tqdm(dataloader):
            output = self.forward(data['features'])
            accuracy = self.net_accuracy(output, data['labels'])
            mean_accuracy.append(np.mean(accuracy))
        

        torch.cuda.empty_cache()
        return {"loss": loss, "accuracy": np.mean(mean_accuracy), "epoch": epoch}
        
# modified network module for self supervised representation learning using ECG
class SelfSupervisedNet2(nn.Module):
    
    def __init__(self, device=torch.device("cpu"), config=[]):
        super(SelfSupervisedNet2, self).__init__()
        
        # import configurations
        self.config = config

        self.num_flat_feat = config['torch']['SSL_flat_feat']
        self.device = device
        self.writer = SummaryWriter(log_dir=self.config['torch']['SSL_runs'] + "run2_" +
                                            str(len(os.listdir(self.config['torch']['SSL_runs'])))
                                            if os.path.exists(self.config['torch']['SSL_runs']) else 0)

        # convolutional layers
        self.conv_layer1 = self.conv_block(1, 32, 32, 1, 0)
        
        self.conv_layer2 = self.conv_block(32, 64, 16, 1, 0)

        self.conv_layer3 = self.conv_block(64, 128, 8, 1, 0)

        # fully connected layers for each task
        self.task = self.fully_connect_block(self.num_flat_feat, 7, 0.5)

        # weight initialization of network
        self.weight_init()

        # objective function
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input 
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        conv = nn.Sequential(
            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob)
        )

        return conv

    def fully_connect_block(self, in_features, out_features, dropout_prob):
        
        dense = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=128, out_features=out_features),
            nn.PReLU(),
            # nn.Dropout(p=dropout_prob),

            # nn.Sigmoid()
        )

        return dense

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):
        # self-supervised network
        x = self.conv_layer1(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer2(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer3(x)
        x = F.max_pool1d(x, kernel_size=x.size()[-1]) # global max pooling
        # x = torch.amax(x, dim=-1) # alternative way of max pooling

        # flatten the output
        x = x.view(-1, self.num_flat_feat)

        # task-specific fully connected layers for each signal transformations
        x = self.task(x)
        
        return x
    
    def total_loss(self, output, labels):
        
        target = labels.to(self.device, dtype=torch.long).reshape(-1, )
        loss = self.loss_criterion(output, target)
        
        return loss.to(self.device)
    
    def net_accuracy(self, output, labels):
        
        target = labels.detach().to('cpu').numpy().reshape(-1, )
        prediction  = torch.argmax(output, dim=1).detach().to('cpu').numpy()
        
        total = len(target) 
        correct = (prediction == target).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def train_model(self, train_dataloader, validation_dataloader, optimizer, scheduler, batch_size, epochs, task_weights):
        model_save_path = self.config['torch']['SSL_models'] + 'model2_' + str(len(os.listdir(self.config['torch']['SSL_models'])))
        utils.makedirs(model_save_path)

        print('training SSL model2...')
        for epoch in range(epochs):
            self.train(mode=True)

            running_loss = 0
            counter = 0
            for i, data in enumerate(train_dataloader):
                optimizer.zero_grad()

                output = self.forward(data['features'].to(self.device, dtype=torch.float))
    
                self.loss   = self.total_loss(output, data['labels'])

                self.loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                optimizer.step()

                running_loss += self.loss.item()
                counter = i + 1
            
            self.eval()

            train_accuracy = self.net_accuracy(output, data['labels'])

            print('Device: {}, Loss: {}, Accuracy: {} at epoch: {}'.format(self.device, running_loss/counter, train_accuracy, epoch+1))

            # Validation 
            valid_accuracy = []
            for valid_data in validation_dataloader:
                valid_output = self.forward(valid_data['features'].to(self.device)) # copy the validation data to cuda
                valid_data['features'].to(torch.device('cpu')) # remove the validation data from cuda
                # print(torch.cuda.memory_summary(device=self.device))
                temp_accuracy = self.net_accuracy(valid_output, valid_data['labels'])
                valid_accuracy.append(temp_accuracy)

            self.writer.add_scalar('Training Loss', running_loss/counter, epoch+1)
            self.writer.add_scalar('Training Accuracy', train_accuracy, epoch+1)
            self.writer.add_scalar('Validation Accuracy', np.mean(valid_accuracy), epoch+1)

            running_loss = 0
            if scheduler:
                scheduler.step()
           
            # save torch model
            if epoch % 100 == 99:
                # (str(len(os.listdir(self.config['torch']['SSL_models']))) if os.path.exists(self.config['torch']['SSL_models']) else 0) +
                model_path = model_save_path + '/net_' + str(epoch+1) + '.pth' 
                torch.save({ 'epoch': epoch+1,
                             'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': self.loss,
                            }, model_path)

        self.writer.close()
        print('Finished training')

    def validate_model(self, dataloader, optimizer, checkpoint, device=torch.device('cpu')):

        # load the model
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss  = checkpoint['loss']

        self.eval()
        
        mean_accuracy = []
        for data in tqdm(dataloader):
            output = self.forward(data['features'])
            accuracy = self.net_accuracy(output, data['labels'])
            mean_accuracy.append(np.mean(accuracy))
        

        torch.cuda.empty_cache()
        return {"loss": loss, "accuracy": np.mean(mean_accuracy), "epoch": epoch}


##########################################################################
# ------ Downstream Emotion recognition architectures ------ #
##########################################################################
# Network for emotion recognition using ECG
class EcgNet(nn.Module):
    
    def __init__(self, load_model=False, checkpoint=None, device=torch.device("cpu"), config=[]):
        super(EcgNet, self).__init__()

        # import configurations
        self.config = config

        self.num_flat_feat = config['torch']['EMOTION_flat_feat']
        self.device = device

        # fully connected layers for recognizing arousal and valence levels
        self.dense_net = self.fully_connect_block(self.num_flat_feat, 2, 0)

        # weight initialization of network
        self.weight_init()

        # objective function
        self.loss_criterion = nn.MSELoss(reduction='sum')
        # self.loss_criterion = nn.L1Loss(reduction='sum')
        # self.loss_criterion = nn.SmoothL1Loss(reduction='sum')

        # load the model
        if load_model:
            self.load_model_from_dict(checkpoint)
    
    def load_model_from_dict(self, checkpoint):
        """ Load the model from the checkpoint by filtering out the unnecessary parameters"""
        model_dict = self.state_dict()
        # filter out unnecessary keys in the imported model
        pretrained_dict = {k:v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}

        # overwrite the entries in the existing state dictionary
        model_dict.update(pretrained_dict)
        
        # load the new state dict
        self.load_state_dict(model_dict)

    def fully_connect_block(self, in_features, out_features, dropout_prob):
        
        dense = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=128, out_features=out_features)
            # nn.PReLU(),
            # nn.Dropout(p=dropout_prob), # dropout is not generally used on the output layer

            # we are removing the sigmoid activation to perform regression
            # nn.Sigmoid()
        )

        return dense

    def weight_init(self):
        """Initialize the weights of the network"""
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

    def forward(self, x, operation='train'):
        # self-supervised network
        # x = self.conv_layer1(x)
        # x = F.max_pool1d(x, kernel_size=8, stride=2)

        # x = self.conv_layer2(x)
        # x = F.max_pool1d(x, kernel_size=8, stride=2)

        # x = self.conv_layer3(x)        
        # x = F.max_pool1d(x, kernel_size=x.size()[-1]) # global max pooling
        # x = torch.amax(x, dim=-1) # alternative way of max pooling

        # flatten the output
        # x = x.view(-1, self.num_flat_feat)

        # fully connected layers for recognizing arousal and valence levels
        x = self.dense_net(x)
        
        return x
    
    def total_loss(self, output, labels):
        
        target = labels.to(self.device, dtype=torch.float).reshape(-1, 2)
        loss   = self.loss_criterion(output, target)

        return loss.to(self.device)
    
    def net_accuracy(self, output, labels):
        target = np.round(labels.numpy().reshape(-1, 2))
        prediction  = np.round(output.numpy())
        
        total = np.max(target.shape) # multi output
        
        # print(prediction, target)
        correct_arousal = (prediction[:, 0] == target[:, 0]).sum().item()
        correct_valence = (prediction[:, 1] == target[:, 1]).sum().item()

        arousal_accuracy = 100 * correct_arousal / total
        valence_accuracy = 100 * correct_valence / total

        return [arousal_accuracy, valence_accuracy]

    def mse_r2score(self, output, labels):
        target = labels.numpy().reshape(-1, 2)
        prediction = output.numpy()

        ars_mse = mean_squared_error(target[:, 0], prediction[:, 0])
        ars_r2  = r2_score(target[:, 0], prediction[:, 0])

        val_mse = mean_squared_error(target[:, 1], prediction[:, 1])
        val_r2  = r2_score(target[:, 1], prediction[:, 1])

        return ars_mse, ars_r2, val_mse, val_r2

    def three_dim_accuracy(self, output, labels):
        """Use Arousal-Valence dimensional model (2d plane) for calculating 
        the accuracy labels assigned in anticlockwise direction
        quadrant1 : class1, quadrant2: class2, quadrant3 and 4: class3"""
        target = np.round(labels.numpy().reshape(-1, 2))
        prediction = np.round(output.numpy())

        ind = np.arange(target.shape[0])
        modified_target = np.zeros((target.shape[0], 1))
        modified_preds  = np.zeros((target.shape[0], 1))

        modified_target[ind[(target[:,0]>2.5) & (target[:,1]>=3.5)]]  = 1
        modified_target[ind[(target[:,0]>2.5) & (target[:,1]<3.5)]]  = 2
        modified_target[ind[(target[:,0]<=2.5)]] = 3

        modified_preds[ind[(prediction[:,0]>2.5) & (prediction[:,1]>=3.5)]]  = 1
        modified_preds[ind[(prediction[:,0]>2.5) & (prediction[:,1]<3.5)]]  = 2
        modified_preds[ind[(prediction[:,0]<=2.5)]] = 3

        total = np.max(target.shape) # multi output
        
        # print(prediction, target)
        correct = (modified_preds == modified_target).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def train_model(self, train_dataloader, validation_dataloader, optimizer, scheduler, batch_size, load_model, use_model, epochs):

        self.writer = SummaryWriter(log_dir=self.config['torch']['EMOTION_runs'] + "run" + use_model + "_" +
                                        str(len(os.listdir(self.config['torch']['EMOTION_runs'])))
                                        if os.path.exists(self.config['torch']['EMOTION_runs']) else 0)

        model_save_path = self.config['torch']['EMOTION_models'] + "model" + use_model + "_" + str(len(os.listdir(self.config['torch']['EMOTION_models'])))
        utils.makedirs(model_save_path)

        _, ax = plt.subplots(1, 2)
        for epoch in range(epochs):
            ax[0].cla()
            ax[1].cla()
            running_loss = 0
            counter = 0 
            arousal_train, valence_train = [], []
            train_3dim_accuracy = []

            for i, data in enumerate(train_dataloader):
                # Make sure that the model is set to training mode (useful to consider Dropout and Batchnorm)
                self.train(mode=True)

                optimizer.zero_grad()

                output = self.forward(data['features'].to(self.device, dtype=torch.float))

                self.loss   = self.total_loss(output, data['labels'])

                self.loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

                running_loss += self.loss.item()
                counter = i + 1

                ars, val = self.net_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu'))
                arousal_train.append(ars)
                valence_train.append(val)
                train_3dim_accuracy.append(self.three_dim_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu')))
               
                ax[0].plot(output.detach().to('cpu').numpy()[:, 1], output.detach().to('cpu').numpy()[:, 0], 'r.')
                ax[0].plot(data['labels'].detach().to('cpu')[:, 1], data['labels'].detach().to('cpu')[:, 0], 'b.')

                # self.mse_r2score(output.detach().to('cpu'), data['labels'].detach().to('cpu'))

            print('Loss: {0:.3f}, Arousal : {1:.3f}, Valence : {2:.3f}, 3-Dim Accuracy: {3:.3f}, at epoch: {4:d}'.format(running_loss/counter, np.mean(arousal_train), np.mean(valence_train), np.mean(train_3dim_accuracy), epoch+1))
            # print('Loss: {0:.3f}, Arousal : {1:.3f}, Valence : {2:.3f} at epoch: {3:d}'.format(running_loss/counter, np.mean(arousal_train), np.mean(valence_train), epoch+1))

            # Validation the model 
            self.eval() # equivalent to self.train(mode=False) (disregards Dropout and Batchnorm during testing the network)

            arousal_valid, valence_valid = [], []
            valid_3dim_accuracy= []
            for valid_data in validation_dataloader:
                valid_output = self.forward(valid_data['features'].to(self.device)) # copy the validation data to cuda
                valid_data['features'].to(torch.device('cpu')) # remove the validation data from cuda
                # print(torch.cuda.memory_summary(device=self.device))
                
                arousal_acc, valence_acc  = self.net_accuracy(valid_output.detach().to('cpu'), valid_data['labels'].detach().to('cpu'))
                val1_accuracy = self.three_dim_accuracy(valid_output.detach().to('cpu'), valid_data['labels'].detach().to('cpu'))
                
                arousal_valid.append(arousal_acc)
                valence_valid.append(valence_acc)
                valid_3dim_accuracy.append(val1_accuracy)

                ax[1].plot(valid_output.detach().to('cpu').numpy()[:, 1], valid_output.detach().to('cpu').numpy()[:, 0], 'r.')
                ax[1].plot(valid_data['labels'].detach().to('cpu')[:, 1], valid_data['labels'].detach().to('cpu')[:, 0], 'b.')
            

            self.writer.add_scalar('Training Loss', running_loss/counter, epoch+1)
            self.writer.add_scalar('Arousal Training Accuracy', np.mean(arousal_train), epoch+1)
            self.writer.add_scalar('Valence Training Accuracy', np.mean(valence_train), epoch+1)
            self.writer.add_scalar('training 3-dim accuracy', np.mean(train_3dim_accuracy), epoch+1)

            self.writer.add_scalar('Arousal Validation Accuracy', np.mean(arousal_valid), epoch+1)
            self.writer.add_scalar('Valence Validation Accuracy', np.mean(valence_valid), epoch+1)
            self.writer.add_scalar('Validation 3-dim Accuracy', np.mean(valid_3dim_accuracy), epoch+1)

            running_loss = 0
            if scheduler:
                    scheduler.step()

            # save torch model
            if epoch % 100 == 99:
                
                # (str(len(os.listdir(self.config['torch']['SSL_models']))) if os.path.exists(self.config['torch']['SSL_models']) else 0) +
                model_path = model_save_path + '/net_' + str(epoch+1) + '.pth' 

                torch.save({ 'epoch': epoch+1,
                             'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': self.loss,
                            }, model_path)
            
            if epoch % 50 == 49:
                ax[0].set_xlim([1, 6])
                ax[0].set_ylim([1, 6])
                ax[1].set_xlim([1, 6])
                ax[1].set_ylim([1, 6])
                plt.suptitle('EcgNet')
                plt.pause(.01)
                
        self.writer.close()
        print('Finished training')

    def validate_model(self, dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=[]):

        # load the new state dict
        self.load_model_from_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss  = checkpoint['loss']

        self.eval()

        mean_arousal_accuracy, mean_valence_accuracy, mse, r2score = [], [], [], []
        # mean_accuracy_3dim, mean_accuracy_4dim = [], []
        for data in tqdm(dataloader):
            output = self.forward(data['features'])

            arousal_score, valence_score = self.net_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu'))

            mean_arousal_accuracy.append(arousal_score)
            mean_valence_accuracy.append(valence_score)

            mse.append(mean_squared_error(data['labels'].detach().to('cpu').numpy(), output.detach().to('cpu').numpy().reshape(-1, 2)))
            r2score.append(r2_score(data['labels'].detach().to('cpu').numpy(), output.detach().to('cpu').numpy().reshape(-1, 2)))

            ax.plot(output.detach().to('cpu').numpy()[:, 1], output.detach().to('cpu').numpy()[:, 0], 'r.')
            ax.plot(data['labels'].detach().to('cpu')[:, 1], data['labels'].detach().to('cpu')[:, 0], 'b.')
        
        ax.set_xlim([1, 6])
        ax.set_ylim([1, 6])
        # ax.grid() 
        # plt.title('')

        torch.cuda.empty_cache()
        return {"loss"   : loss, 
                "Arousal": np.mean(mean_arousal_accuracy),
                "Valence": np.mean(mean_valence_accuracy), 
                "mse"    : np.mean(mse), 
                "r2score": np.mean(r2score), 
                "epoch"  : epoch}

    def predict(self, dataloader, checkpoint):
        '''Return N samples x 2 array of predictions made by the EmotionNet'''
        # load the model
        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()

        predictions = []
        for data in tqdm(dataloader):
            output = self.forward(data['features'].to(self.device, dtype=torch.float))
            predictions.append(output.detach().to('cpu').numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        return predictions

# Multi-modal EMOTION recognition net 
class EmotionNet(nn.Module):

    def __init__(self, num_feats, device=torch.device("cpu"), config=[]):
        super(EmotionNet, self).__init__()

        # import configurations
        self.config = config

        self.num_feat = num_feats
        # self.modality = modality # which modality to use for the network ('ecg', 'emg', 'gsr', 'ppg', 'rsp')
        self.device = device

        # fully connected layers for recognizing arousal and valence levels
        if self.num_feat > 0:
            self.dense_net = self.fully_connect_block(self.num_feat, 2, 0.25)
            # self.dense_net = self.fully_connect_block(self.num_feat, 2, 0.1)

            # weight initialization of network
            self.weight_init()

            # objective function
            self.loss_criterion = nn.MSELoss(reduction='sum')
            # self.loss_criterion = nn.L1Loss(reduction='sum')
            # self.loss_criterion = nn.SmoothL1Loss(reduction='sum')


    def fully_connect_block(self, in_features, out_features, dropout_prob):
        max_features = 512
        dense = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=max_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=max_features, out_features=max_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=max_features, out_features=256),
            nn.ReLU(),
            # nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # nn.Dropout(p=dropout_prob),

            nn.Linear(in_features=128, out_features=out_features),
            # nn.ReLU()

        )

        return dense

    def weight_init(self):
        """Initialize the weights of the network"""
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):
        # fully connected layers for recognizing arousal and valence levels
        x = self.dense_net(x)
        
        return x
    
    def total_loss(self, output, labels):
        target = labels.to(self.device, dtype=torch.float).reshape(-1, 2)
        loss   = self.loss_criterion(output, target)

        return loss.to(self.device)
    
    def net_accuracy(self, output, labels):
        target = np.round(labels.numpy().reshape(-1, 2))
        prediction  = np.round((output.numpy()))
        
        total = np.max(target.shape) # multi output
        
        # print(prediction, target)
        correct_arousal = (prediction[:, 0] == target[:, 0]).sum().item()
        correct_valence = (prediction[:, 1] == target[:, 1]).sum().item()

        arousal_accuracy = 100 * correct_arousal / total
        valence_accuracy = 100 * correct_valence / total

        return [arousal_accuracy, valence_accuracy]

    def mse(self, output, labels):
        target = labels.numpy().reshape(-1, 2)
        prediction = output.numpy()

        ars_mse = mean_squared_error(target[:, 0], prediction[:, 0])
        # ars_r2  = r2_score(target[:, 0], prediction[:, 0])

        val_mse = mean_squared_error(target[:, 1], prediction[:, 1])
        # val_r2  = r2_score(target[:, 1], prediction[:, 1])

        return ars_mse, val_mse

    def absolute_error(self, output, labels):
        target = labels.numpy().reshape(-1, 2)
        prediction = output.numpy()

        ars_ae = np.abs(target[:, 0] - prediction[:, 0])
        val_ae = np.abs(target[:, 1] - prediction[:, 1])

        return ars_ae, val_ae

    def three_dim_accuracy(self, output, labels):
        """Use Arousal-Valence dimensional model (2d plane) for calculating 
        the accuracy labels assigned in anticlockwise direction
        quadrant1 : class1, quadrant2: class2, quadrant3 and 4: class3"""
        target = labels.numpy().reshape(-1, 2)
        prediction = output.numpy()

        ind = np.arange(target.shape[0])
        modified_target = np.zeros((target.shape[0], 1))
        modified_preds  = np.zeros((target.shape[0], 1))

        modified_target[ind[(target[:,0]>=2.8) & (target[:,1]>=4.)]]  = 1
        modified_target[ind[(target[:,0]>=2.8) & (target[:,1]<4.)]]  = 2
        modified_target[ind[(target[:,0]<2.8) & (target[:,1]>=5.)]]  = 1
        modified_target[ind[(target[:,0]<2.8) & (target[:,1]<5.) & (target[:,1]>=3.)]] = 3
        modified_target[ind[(target[:,0]<2.8) & (target[:,1]<3.)]]  = 2

        modified_preds[ind[(prediction[:,0]>=2.8) & (prediction[:,1]>=4.)]]  = 1
        modified_preds[ind[(prediction[:,0]>=2.8) & (prediction[:,1]<4.)]]  = 2
        modified_preds[ind[(prediction[:,0]<2.8) & (prediction[:,1]>=5.)]]  = 1
        modified_preds[ind[(prediction[:,0]<2.8) & (prediction[:,1]<5.) & (prediction[:,1]>=3.)]] = 3
        modified_preds[ind[(prediction[:,0]<2.8) & (prediction[:,1]<3.)]]  = 2

        total = np.max(target.shape) # multi output
        
        # print(prediction, target)
        correct = (modified_preds == modified_target).sum().item()

        accuracy = 100 * correct / total

        return accuracy

    def train_model(self, train_dataloader, validation_dataloader, optimizer, scheduler, batch_size, epochs):

        self.writer = SummaryWriter(log_dir=self.config['torch']['EMOTION_runs'] + "MultiModal_run" + "_" +
                                        str(len(os.listdir(self.config['torch']['EMOTION_runs'])))
                                        if os.path.exists(self.config['torch']['EMOTION_runs']) else 0)

        model_save_path = self.config['torch']['EMOTION_models'] + "MultiModal_model" + "_" + str(len(os.listdir(self.config['torch']['EMOTION_models'])))
        utils.makedirs(model_save_path)

        error_thres = 0.7
        color0 = [0., 1., 0.]  # a_err < 0.7 and v_err < 0.7
        color1 = [1., 1., 0.]  # a_err < 0.7 or v_err < 0.7
        color2 = [1., 0., 0.]  # a_err >= 0.7 and v_err >= 0.7

        _, ax = plt.subplots(1, 2)
        for epoch in range(epochs):
            self.train(mode=True)

            ax[0].cla()
            ax[1].cla()

            running_loss = 0
            counter = 0 
            arousal_train, valence_train = [], []
            train_3dim_accuracy = []

            for i, data in enumerate(train_dataloader):
                # Make sure that the model is set to training mode (useful to consider Dropout and Batchnorm)
                self.train(mode=True)

                optimizer.zero_grad()

                output = self.forward(data['features'].to(self.device, dtype=torch.float))
                
                self.loss   = self.total_loss(output, data['labels'])

                self.loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()

                running_loss += self.loss.item()
                counter = i + 1

                # ars, val = self.net_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu'))
                ars, val = self.absolute_error(output.detach().to('cpu'), data['labels'].detach().to('cpu'))
                arousal_train.append(ars)
                valence_train.append(val)
                train_3dim_accuracy.append(self.three_dim_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu')))

                output_cpu = output.detach().to('cpu').numpy()
                labels_cpu = data['labels'].detach().to('cpu')
                ind = np.arange(0, labels_cpu.shape[0])
                ind_small_da = set(ind[ars < error_thres])
                ind_small_dv = set(ind[val < error_thres])
                ind_large_da = set(ind.tolist()).difference(ind_small_da)
                ind_large_dv = set(ind.tolist()).difference(ind_small_dv)
                ind0 = list(ind_small_da.intersection(ind_small_dv))
                ind2 = list(ind_large_da.intersection(ind_large_dv))
                ind1 = list(set(ind.tolist()).difference(ind0).difference(ind2))
                ax[0].plot(output_cpu[ind2, 1], output_cpu[ind2, 0], '.', color=color2, label='Predicted (da>0.7 and dv>0.7)')
                ax[0].plot(output_cpu[ind1, 1], output_cpu[ind1, 0], '.', color=color1, label='Predicted (da<0.7 or dv<0.7)')
                ax[0].plot(output_cpu[ind0, 1], output_cpu[ind0, 0], '.', color=color0, label='Predicted (da<0.7 and dv<0.7)')
                ax[0].plot(labels_cpu[:, 1], labels_cpu[:, 0], 'b.', label='True')

            # Validation the model 
            self.eval() # equivalent to self.train(mode=False) (disregards Dropout and Batchnorm during testing the network)

            arousal_valid, valence_valid = [], []
            valid_3dim_accuracy = []
            for valid_data in validation_dataloader:
                valid_output = self.forward(valid_data['features'].to(self.device)) # copy the validation data to cuda

                valid_data['features'].to(torch.device('cpu')) # remove the validation data from cuda
                # print(torch.cuda.memory_summary(device=self.device))

                # arousal_acc, valence_acc  = self.net_accuracy(valid_output.detach().to('cpu'), valid_data['labels'].detach().to('cpu'))
                arousal_ae, valence_ae = self.absolute_error(valid_output.detach().to('cpu'), valid_data['labels'].detach().to('cpu'))
                val1_accuracy = self.three_dim_accuracy(valid_output.detach().to('cpu'), valid_data['labels'].detach().to('cpu'))

                arousal_valid.append(arousal_ae)
                valence_valid.append(valence_ae)
                valid_3dim_accuracy.append(val1_accuracy)

                output_cpu = valid_output.detach().to('cpu').numpy()
                labels_cpu = valid_data['labels'].detach().to('cpu')
                ind = np.arange(0, labels_cpu.shape[0])
                ind_small_da = set(ind[arousal_ae < error_thres])
                ind_small_dv = set(ind[valence_ae < error_thres])
                ind_large_da = set(ind.tolist()).difference(ind_small_da)
                ind_large_dv = set(ind.tolist()).difference(ind_small_dv)
                ind0 = list(ind_small_da.intersection(ind_small_dv))
                ind2 = list(ind_large_da.intersection(ind_large_dv))
                ind1 = list(set(ind.tolist()).difference(ind0).difference(ind2))
                ax[1].plot(output_cpu[ind2, 1], output_cpu[ind2, 0], '.', color=color2, label='Predicted (da>0.7 and dv>0.7)')
                ax[1].plot(output_cpu[ind1, 1], output_cpu[ind1, 0], '.', color=color1, label='Predicted (da<0.7 or dv<0.7)')
                ax[1].plot(output_cpu[ind0, 1], output_cpu[ind0, 0], '.', color=color0, label='Predicted (da<0.7 and dv<0.7)')
                ax[1].plot(labels_cpu[:, 1], labels_cpu[:, 0], 'b.', label='True')

            arousal_train = np.concatenate(arousal_train, axis=0)
            valence_train = np.concatenate(valence_train, axis=0)

            arousal_small_train = 100. * float(np.extract(arousal_train < error_thres, arousal_train).shape[0]) / float(arousal_train.shape[0])
            valence_small_train = 100. * float(np.extract(valence_train < error_thres, valence_train).shape[0]) / float(valence_train.shape[0])
            both_small_train = 100. * float(np.extract(np.logical_and(arousal_train < error_thres, valence_train < error_thres), arousal_train).shape[0]) / float(arousal_train.shape[0])

            arousal_valid = np.concatenate(arousal_valid, axis=0)
            valence_valid = np.concatenate(valence_valid, axis=0)

            arousal_small_valid = 100. * float(np.extract(arousal_valid < error_thres, arousal_valid).shape[0]) / float(arousal_valid.shape[0])
            valence_small_valid = 100. * float(np.extract(valence_valid < error_thres, valence_valid).shape[0]) / float(valence_valid.shape[0])
            both_small_valid = 100. * float(np.extract(np.logical_and(arousal_valid < error_thres, valence_valid < error_thres), arousal_valid).shape[0]) / float(arousal_valid.shape[0])

            self.writer.add_scalar('Training Loss', running_loss/counter, epoch+1)
            self.writer.add_scalar('Arousal Training error', np.mean(arousal_train), epoch+1)
            self.writer.add_scalar('Valence Training error', np.mean(valence_train), epoch+1)
            self.writer.add_scalar('Arousal Training error < thres', arousal_small_train, epoch+1)
            self.writer.add_scalar('Valence Training error < thres', valence_small_train, epoch+1)
            self.writer.add_scalar('Both Training error < thres', both_small_train, epoch+1)
            self.writer.add_scalar('training 3-dim accuracy', np.mean(train_3dim_accuracy), epoch+1)

            self.writer.add_scalar('Arousal Validation error', np.mean(arousal_valid), epoch+1)
            self.writer.add_scalar('Valence Validation error', np.mean(valence_valid), epoch+1)
            self.writer.add_scalar('Arousal Validation error < thres', arousal_small_valid, epoch+1)
            self.writer.add_scalar('Valence Validation error < thres', valence_small_valid, epoch+1)
            self.writer.add_scalar('Both Validation error < thres', both_small_valid, epoch+1)
            self.writer.add_scalar('Validation 3-dim accuracy', np.mean(valid_3dim_accuracy), epoch+1)

            print('[{6:d}] Loss: {0:.3f}, Train [A: {1:.3f}, A<{7:.1f}: {8:.3f}, V: {2:.3f}, V<{7:.1f}: {9:.3f}, AV<{7:.1f}: {13:.3f}, 3dim: {12:.3f}], Valid [A: {3:.3f}, A<{7:.1f}: {10:.3f}, V: {4:.3f}, V<{7:.1f}: {11:.3f}, AV<{7:.1f}: {14:.3f}, 3dim: {5:.3f}]'.format(running_loss/counter, np.mean(arousal_train), np.mean(valence_train), np.mean(arousal_valid), np.mean(valence_valid), np.mean(valid_3dim_accuracy), epoch+1, error_thres, arousal_small_train, valence_small_train, arousal_small_valid, valence_small_valid, np.mean(train_3dim_accuracy), both_small_train, both_small_valid))
            
            running_loss = 0
            if scheduler:
                scheduler.step()
           
            # save torch model
            if epoch % 100 == 99:
                model_path = model_save_path + '/net_' + str(epoch+1) + '.pth' 

                torch.save({ 'epoch': epoch+1,
                             'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': self.loss,
                            }, model_path)
            
            if epoch % 50 == 49:
                ax[0].set_xlim([1, 6])
                ax[0].set_ylim([1, 6])
                ax[1].set_xlim([1, 6])
                ax[1].set_ylim([1, 6])
                plt.suptitle('DNN')
                plt.pause(.01)

        # The following makes sure that the True labels (blue color) is superimposed on the top of the predicted labels
        for data in train_dataloader:
            ax[0].plot(data['labels'].detach().to('cpu')[:, 1], data['labels'].detach().to('cpu')[:, 0], 'b.', label='True')

        for valid_data in validation_dataloader:
            ax[1].plot(valid_data['labels'].detach().to('cpu')[:, 1], valid_data['labels'].detach().to('cpu')[:, 0], 'b.')

        self.writer.close()
        print('Finished training')
        
        
        # return the arousal and valence absolute errors 
        return arousal_train, valence_train, arousal_valid, valence_valid

    def validate_model(self, dataloader, optimizer, checkpoint, device=torch.device('cpu'), ax=[]):

        # load the model
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss  = checkpoint['loss']

        self.eval()

        arousal_ae, valence_ae = [], []
        # mean_accuracy_3dim, mean_accuracy_4dim = [], []
        for data in tqdm(dataloader):
            output = self.forward(data['features'].to(self.device, dtype=torch.float))

            # arousal_score, valence_score = self.net_accuracy(output.detach().to('cpu'), data['labels'].detach().to('cpu'))
            ars, val = self.absolute_error(output.detach().to('cpu'), data['labels'].detach().to('cpu'))

            arousal_ae.append(ars)
            valence_ae.append(val)

            if ax:
                # ax.plot(output.detach().to('cpu').numpy()[:, 1], output.detach().to('cpu').numpy()[:, 0], 'r.')
                # ax.plot(data['labels'].detach().to('cpu')[:, 1], data['labels'].detach().to('cpu')[:, 0], 'b.')
            
                ind = np.arange(0, data['labels'].detach().to('cpu').shape[0])
                err = ars + val
                ind1 = ind[err <= 1]
                color1 = [0, 0, 1]

                ind2 = ind[(err > 1) & (err <= 2)]
                color2 = [0, 1, 0]
                
                ind3 = ind[err > 2]
                color3 = [1, 0, 0]

                ax.plot(data['labels'].detach().to('cpu')[ind1, 1], data['labels'].detach().to('cpu')[ind1, 0], '.', color=color1)
                ax.plot(data['labels'].detach().to('cpu')[ind2, 1], data['labels'].detach().to('cpu')[ind2, 0], '.', color=color2)
                ax.plot(data['labels'].detach().to('cpu')[ind3, 1], data['labels'].detach().to('cpu')[ind3, 0], '.', color=color3)


        # if ax:
        #     for data in dataloader:
        #         ax.plot(data['labels'].detach().to('cpu')[:, 1], data['labels'].detach().to('cpu')[:, 0], 'b.')

        arousal_ae = np.concatenate(arousal_ae, axis=0)
        valence_ae = np.concatenate(valence_ae, axis=0)

        torch.cuda.empty_cache()
        return {"loss": loss, 
                "Arousal": arousal_ae,
                "Valence": valence_ae, 
                "epoch": epoch}
    
    def predict(self, dataloader, checkpoint):
        '''Return N samples x 2 array of predictions made by the EmotionNet'''
        # load the model
        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()

        predictions = []
        for data in tqdm(dataloader):
            output = self.forward(data['features'].to(self.device, dtype=torch.float))
            predictions.append(output.detach().to('cpu').numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        return predictions


# Multi-modal EMOTION recognition using raw data and Conv1d
class EmotionNetConv1d(EmotionNet):

    def __init__(self, window_length, num_in_channels=1, device=torch.device("cpu"), config=[]):
        super(EmotionNetConv1d, self).__init__(num_feats=0, device=device, config=config)

        self.window_length = window_length
        self.num_in_channels = num_in_channels

        # self.network = 'small'
        self.network = 'large'
        # self.network = 'larger'  # NG

        if self.network == 'small':
            self.num_conv_out_channels = 64
            self.max_fully_connect_size = 128
            self.conv_layer1 = self.conv_block(self.num_in_channels, 32, 16, 4, 0.1)
            self.conv_layer2 = self.conv_block(32, self.num_conv_out_channels, 8, 2, 0.1, last_layer=True)
            self.conv_layer3 = None
            self.dense_net = self.fully_connect_block(self.num_conv_out_channels, 2, 0.25)

        elif self.network == 'large':
            self.num_conv_out_channels = 128
            self.max_fully_connect_size = 256
            self.conv_layer1 = self.conv_block(self.num_in_channels, 32, 16, 4, 0.1)
            self.conv_layer2 = self.conv_block(32, 64, 8, 2, 0.1)
            self.conv_layer3 = self.conv_block(64, self.num_conv_out_channels, 4, 1, 0.1, last_layer=True)
            self.dense_net = self.fully_connect_block(self.num_conv_out_channels, 2, 0.25)

        elif self.network == 'larger':
            self.num_conv_out_channels = 256
            self.max_fully_connect_size = 256
            self.conv_layer1 = self.conv_block(self.num_in_channels, 64, 16, 4, 0.1)
            self.conv_layer2 = self.conv_block(64, 128, 8, 2, 0.1)
            self.conv_layer3 = self.conv_block(128, self.num_conv_out_channels, 4, 1, 0.1, last_layer=True)
            self.dense_net = self.fully_connect_block(self.num_conv_out_channels, 2, 0.25)

        # weight initialization of network
        self.weight_init()

        self.loss_criterion = nn.MSELoss(reduction='sum')


    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob, last_layer=False):
        # pad the layers such that the output has the same size of input
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        layers = OrderedDict()
        layers['pad0'] = nn.ConstantPad1d(padding=(pad_left, pad_right), value=0)
        layers['conv0'] = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        layers['relu0'] = nn.LeakyReLU()
        layers['dropout0'] = nn.Dropout(p=dropout_prob)

        layers['pad1'] = nn.ConstantPad1d(padding=(pad_left, pad_right), value=0)
        layers['conv1'] = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        layers['relu1'] = nn.LeakyReLU()
        layers['dropout1'] = nn.Dropout(p=dropout_prob)

        if last_layer:
            # layers['pool'] = nn.AdaptiveAvgPool1d(1)
            layers['pool'] = nn.AdaptiveMaxPool1d(1)
        else:
            # layers['pool'] = nn.AvgPool1d(kernel_size=8, stride=2)
            layers['pool'] = nn.MaxPool1d(kernel_size=4, stride=2)

        conv = nn.Sequential(layers)
        return conv


    def fully_connect_block(self, in_features, out_features, dropout_prob):
        layers = OrderedDict()
        layers['dense0'] = nn.Linear(in_features=in_features, out_features=self.max_fully_connect_size)
        layers['relu0'] = nn.ReLU()
        layers['dropout0'] = nn.Dropout(p=dropout_prob)

        out_size = self.max_fully_connect_size
        i_layer = 1
        while out_size >= 64:
            layers['dense' + str(i_layer)] = nn.Linear(in_features=out_size, out_features=round(out_size/2))
            layers['relu' + str(i_layer)] = nn.ReLU()
            if out_size >= 128:
                layers['dropout' + str(i_layer)] = nn.Dropout(p=dropout_prob)

            out_size = round(out_size / 2)
            i_layer += 1

        # last layer
        layers['dense' + str(i_layer)] = nn.Linear(in_features=out_size, out_features=out_features)

        print('fully_connect_block: number of layers=', i_layer+1)

        dense = nn.Sequential(layers)
        return dense


    def forward(self, x):
        # conv1d layers
        conv_in = x.view(-1, self.num_in_channels, self.window_length)
        conv_out = self.conv_layer1(conv_in)
        # conv_out = F.max_pool1d(conv_out, kernel_size=8, stride=2)
        # conv_out = F.avg_pool1d(conv_out, kernel_size=8, stride=2)
        conv_out = self.conv_layer2(conv_out)
        if self.conv_layer3 != None:
            # conv_out = F.max_pool1d(conv_out, kernel_size=8, stride=2)
            # conv_out = F.avg_pool1d(conv_out, kernel_size=8, stride=2)
            conv_out = self.conv_layer3(conv_out)

        # conv_out = F.max_pool1d(conv_out, kernel_size=conv_out.size()[-1]) # global max pooling
        # conv_out = F.avg_pool1d(conv_out, kernel_size=conv_out.size()[-1]) # global max pooling
        # flatten the output
        conv_out = conv_out.view(-1, self.num_conv_out_channels)
        # dense layer
        pred = self.dense_net(conv_out)
        return pred


# Multi-modal EMOTION recognition using LSTM
class EmotionNetLSTM(EmotionNet):

    def __init__(self, num_feats, seq_len, num_in_channels=1, device=torch.device("cpu"), config=[]):
        super(EmotionNetLSTM, self).__init__(num_feats, device=device, config=config)
        # number of LSTM cells = number of windows in a sequence
        self.seq_len = seq_len
        # number of input/output channels of the Conv1d unit (takes one window)
        self.n_conv_in = num_in_channels
        self.n_conv_out = 32
        conv1d = self.conv_block(in_channels=self.n_conv_in, out_channels=self.n_conv_out, kernel_size=4, stride=1, dropout_prob=0.1)
        # LSTM: each cell takes output of one Conv1d unit
        n_lstm_layers = 1
        self.lstm_hidden_size = 64
        lstm = nn.LSTM(self.n_conv_out, self.lstm_hidden_size, n_lstm_layers, batch_first=True)
        # Fully connected layer: outputs (arousal, valence)
        dense = self.fully_connect_block(self.lstm_hidden_size, 2, 0.1)
        self.dense_net = nn.ModuleDict({
            'conv1d': conv1d,
            'lstm': lstm,
            'dense': dense
            })


    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        layers = OrderedDict()
        layers['pad0'] = nn.ConstantPad1d(padding=(pad_left, pad_right), value=0)
        layers['conv0'] = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        layers['relu0'] = nn.LeakyReLU()
        layers['dropout0'] = nn.Dropout(p=dropout_prob)

        layers['pad1'] = nn.ConstantPad1d(padding=(pad_left, pad_right), value=0)
        layers['conv1'] = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        layers['relu1'] = nn.LeakyReLU()
        layers['dropout1'] = nn.Dropout(p=dropout_prob)

        layers['pool'] = nn.AdaptiveMaxPool1d(1)

        conv = nn.Sequential(layers)
        return conv


    def forward(self, x):
        # x: [seq_len*batch, n_conv_in, num_feat]
        # input x[i, :, :] to the conv1d layer (common for every frame)
        conv_out = self.dense_net['conv1d'](x)
        # conv_out: [seq_len*batch, n_conv_out, 1] -> [batch, seq_len, n_conv_out] (note batch_first=True)
        conv_out = conv_out.view(-1, self.seq_len, self.n_conv_out)
        lstm_output, _ = self.dense_net['lstm'](conv_out)
        #print('lstm output size:', lstm_output.shape)
        #print('lstm output:', lstm_output)
        # [batch, seq_len, lstm_hidden] -> last output [batch, lstm_hidden]
        dense_input = lstm_output[:,-1,:].reshape(-1, self.lstm_hidden_size)
        #print('dense_input size:', dense_input.shape)
        #print('dense_input:', dense_input)
        dense_output = self.dense_net['dense'](dense_input)
        #print('dense_output size:', dense_output.shape)
        #print('dense_output:', dense_output)
        # repeat each row seq_len times to match the size of ground truth label data
        pred = dense_output.repeat(1, self.seq_len).reshape(-1, 2)
        #print('pred size:', pred.shape)
        #print('pred:', pred)
        return pred


##########################################################################
# ------ ECG features from pretrained Self Supervised Network ------ #
##########################################################################
# ECG features from self supervised network
class SelfSupervisedNetFeats(nn.Module):
    
    def __init__(self, load_model=False, checkpoint=None, num_flat_feat=128, device=torch.device("cpu"), config=[]):
        super(SelfSupervisedNetFeats, self).__init__()

        # import configurations
        self.config = config

        self.num_flat_feat = num_flat_feat
        self.device = device

        # convolutional layers
        self.conv_layer1 = self.conv_block(1, 32, 32, 1, 0)
        
        self.conv_layer2 = self.conv_block(32, 64, 16, 1, 0)

        self.conv_layer3 = self.conv_block(64, 128, 8, 1, 0)

        # load the model
        if load_model:

            model_dict = self.state_dict()
            # filter out unnecessary keys in the imported model
            pretrained_dict = {k:v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}

            # overwrite the entries in the existing state dictionary
            model_dict.update(pretrained_dict)
            
            # load the new state dict
            self.load_state_dict(model_dict)

            self.eval()

    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input 
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        conv = nn.Sequential(
            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),

            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob)
        )

        return conv

    def forward(self, x):
        # self-supervised network
        x = self.conv_layer1(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer2(x)
        x = F.max_pool1d(x, kernel_size=8, stride=2)

        x = self.conv_layer3(x)        
        x = F.max_pool1d(x, kernel_size=x.size()[-1]) # global max pooling
        # x = torch.amax(x, dim=-1) # alternative way of max pooling

        # flatten the output
        x = x.view(-1, self.num_flat_feat)
        
        return x
    
    def return_SSL_feats(self, dataloader):
        
        features, labels = [], []
        for data in dataloader:
            output = self.forward(data['features'].to(self.device, dtype=torch.float))
            features.append(output.detach().to('cpu').numpy())
            labels.append(data['labels'].detach().to('cpu').numpy())
            # free the memory
            data['features'].detach().to('cpu')

        features = np.concatenate(features, axis=0)
        labels   = np.concatenate(labels, axis=0)

        return features, labels
 

# test self-supervised network using sample data
if __name__ == '__main__':

    window_size = 100 * config['window_size']

    # create the directories to store the runs and pickle models
    if ~os.path.exists("runs/SSL_runs"):
        utils.makedirs("runs/SSL_runs")

    if ~os.path.exists("models/SSL_models"):
        utils.makedirs("models/SSL_models")

    # Load sample ECG data
    x = np.load(Path(__file__).parents[1] / 'data/sample_ecg.npy')
    x_train = np.concatenate((x[:, :window_size], x[:, window_size:2*window_size]), axis=0).astype(np.float64).reshape(-1, 1, window_size)
    x_test  = x[:, 2*window_size:].astype(np.float64).reshape(-1, 1, 560)

    ## transformation task params
    noise_param = 15 #noise_amount
    scale_param = 1.1 #scaling_factor
    permu_param = 20 #permutation_pieces
    tw_piece_param = 9 #time_warping_pieces
    twsf_param = 1.05 #time_warping_stretch_factor
    batch_size = 10

    # apply transformations to the raw data
    train_ecg   = []
    train_labels = []
    test_ecg   = []
    test_labels = []
    for i in range(x_train.shape[0]):
        signal = x_train[i, :, :].T
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)
        
        train_ecg.append(tr_signal)
        train_labels.append(tr_labels)
   
    for i in range(x_test.shape[0]):
        signal = x_test[i, :, :].T
        tr_signal, tr_labels = sgtf.apply_all_transformations(signal, noise_param, scale_param, permu_param, tw_piece_param, twsf_param, 1/twsf_param)
        
        test_ecg.append(tr_signal)
        test_labels.append(tr_labels)

    train_ecg    = np.concatenate(train_ecg, axis=0).reshape(-1, 1, window_size)
    train_labels = np.concatenate(train_labels, axis=0).reshape(-1, 1)
    test_ecg     = np.concatenate(test_ecg, axis=0).reshape(-1, 1, 560)
    test_labels  = np.concatenate(test_labels, axis=0).reshape(-1, 1)

    train_path   = str(Path(__file__).parents[1] / 'data/train_sample.h5')
    test_path    = str(Path(__file__).parents[1] / 'data/test_sample.h5')

    train_data   = {'ECG': train_ecg, 'labels': train_labels}
    dd.io.save(train_path, train_data)

    test_data   = {'ECG': test_ecg, 'labels': test_labels}
    dd.io.save(test_path, test_data)

    train_dataset = EcgDataset(train_path, window_size)
    test_dataset  = EcgDataset(test_path , window_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # Network
    net = SelfSupervisedNet(device, config)
    net.to(device)

    # see if an exponential decay learning rate scheduler is required
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)

    net.train_model(train_dataloader, test_dataloader, optimizer, scheduler=None, batch_size=batch_size, epochs=15, task_weights=[1, 1, 1, 1, 1, 1, 1])

    
    


