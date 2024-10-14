import torch
import numpy as np
from utils import *
import pandas as pd
from sklearn.metrics import roc_auc_score


def make_prediction(X_CCC_head, X_CCC_body, X_CCO_head, X_CCO_body, X_GCO_head, X_GCO_body, device=torch.device("cpu"),
                    folder="./Train/"):
    ck = folder + "best_checkpoint.pth"
    # load the best model
    model, optimizer, best_epoch = load_model(ck, device)

    with torch.no_grad():
        model.eval()
        C_CC_magnitude_head, C_CC_magnitude_body, C_CO_magnitude_head, C_CO_magnitude_body, G_CO_magnitude_head, \
        G_CO_magnitude_body = model.forward(X_CCC_head, X_CCC_body, X_CCO_head, X_CCO_body, X_GCO_head, X_GCO_body)
        return C_CC_magnitude_head, C_CC_magnitude_body, C_CO_magnitude_head, C_CO_magnitude_body, \
               G_CO_magnitude_head, G_CO_magnitude_body, best_epoch
