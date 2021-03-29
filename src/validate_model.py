import yaml
import os
from pathlib import Path
import torch 
import torch.optim as optim
from NN_models import EcgDataset, SelfSupervisedNet



config = yaml.load(open(Path(__file__).parents[1] / 'config.yml'), Loader=yaml.SafeLoader)

if __name__ == "__main__":

    device = torch.device("cpu")
    
    model = SelfSupervisedNet(device, config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    window_size = config['freq'] * config['window_size']
    train_data = EcgDataset(config['transformed_data'], window_size, data_group='/train')
    valid_data = EcgDataset(config['transformed_data'], window_size, data_group='/valid')

    # train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, shuffle=True) 

    
    for file in os.listdir(config['torch']['SSL_models']):
        checkpoint = torch.load(config['torch']['SSL_models'] + file, map_location=device)
        results = model.validate_model(valid_dataloader, model, optimizer, checkpoint, device=device)

        print("Epoch: {}, Accuracy: {}".format(results['epoch'], results['accuracy']))

    # writer = SummaryWriter(log_dir=)
    # writer.add_scalar(data_name, , epoch+1)
    # writer.close()
