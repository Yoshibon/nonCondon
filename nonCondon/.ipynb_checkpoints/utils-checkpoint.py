import pandas as pd
import torch
from nonCondon.Models import *
from torch.optim import Adam


def load_model(file_path, device):
    checkpoint = torch.load(file_path)
    model = MagnitudeModel(input_dim=checkpoint['input_dim'], dim_hidden=checkpoint['dim_hidden'],
                           layer_num=checkpoint['layer_num'], dropout=checkpoint['dropout'],
                           dim_reg=checkpoint['dim_reg'])
    model.load_state_dict(checkpoint['state_dict'])
    _ = model.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_epoch = checkpoint['epoch']
    return model, optimizer, best_epoch


def save_tensor(tensor_input, file_name):
    df = pd.DataFrame(tensor_input.numpy(), columns=['magnitude'])
    df.to_csv(file_name, index=False)