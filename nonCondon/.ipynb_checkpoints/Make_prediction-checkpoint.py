import torch
import numpy as np
from nonCondon.utils import *
import pandas as pd
import pickle
from os.path import join

def load_standardizers(file_folder):
    with open(join(file_folder, 'sd_CCC_head.pkl'),'rb') as save_file:
        sd_CCC_head = pickle.load(save_file)
    with open(join(file_folder, 'sd_CCO_head.pkl'),'rb') as save_file:
        sd_CCO_head = pickle.load(save_file)
    with open(join(file_folder, 'sd_GCO_head.pkl'),'rb') as save_file:
        sd_GCO_head = pickle.load(save_file)
        
    with open(join(file_folder, 'sd_CCC_body.pkl'),'rb') as save_file:
        sd_CCC_body = pickle.load(save_file)
    with open(join(file_folder, 'sd_CCO_body.pkl'),'rb') as save_file:
        sd_CCO_body = pickle.load(save_file)
    with open(join(file_folder, 'sd_GCO_body.pkl'),'rb') as save_file:
        sd_GCO_body = pickle.load(save_file)
    return sd_CCC_body, sd_CCC_head, sd_CCO_body, sd_CCO_head, sd_GCO_body, sd_GCO_head


def make_prediction(X_CCC_head, X_CCC_body, X_CCO_head, X_CCO_body, X_GCO_head, X_GCO_body, device=torch.device("cpu"),
                    folder="./Train/"):
    ck = join(folder, "best_checkpoint.pth")
    # load the best model
    model, optimizer, best_epoch = load_model(ck, device)

    with torch.no_grad():
        model.eval()
        C_CC_magnitude_head, C_CC_magnitude_body, C_CO_magnitude_head, C_CO_magnitude_body, G_CO_magnitude_head, \
        G_CO_magnitude_body = model.forward(X_CCC_head, X_CCC_body, X_CCO_head, X_CCO_body, X_GCO_head, X_GCO_body)
        return C_CC_magnitude_head, C_CC_magnitude_body, C_CO_magnitude_head, C_CO_magnitude_body, \
               G_CO_magnitude_head, G_CO_magnitude_body
