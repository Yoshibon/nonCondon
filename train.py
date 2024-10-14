import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import Adam

from nonCondon.Models import *
from nonCondon.utils import *
from nonCondon.Make_prediction import *
import nonCondon.simply_A as spA
import nonCondon.simply_B as spB


print("Data Loading Finished")
parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--dim_model', type=int, default=64)
args = parser.parse_args()

lr = args.lr
wd = args.wd
dim_model = args.dim_model
layer_num = args.layer_num
nsnpsht = 200000
save_model_every = 10

# Read Dataset B
true_results_B = pd.read_csv("B-helix.csv", header=None)
true_results_B = true_results_B.values
true_freq_B = torch.FloatTensor(true_results_B[:, 0])
true_inten_B = torch.FloatTensor(true_results_B[:, 1])

X_CCC_body_B = pd.read_csv("Data/B_Helix/electric_field_C_CC_body.dat", header=None, sep='\s+')
X_CCC_head_B = pd.read_csv("Data/B_Helix/electric_field_C_CC_head.dat", header=None, sep='\s+')
X_CCO_body_B = pd.read_csv("Data/B_Helix/electric_field_C_CO_body.dat", header=None, sep='\s+')
X_CCO_head_B = pd.read_csv("Data/B_Helix/electric_field_C_CO_head.dat", header=None, sep='\s+')
X_GCO_body_B = pd.read_csv("Data/B_Helix/electric_field_G_CO_body.dat", header=None, sep='\s+')
X_GCO_head_B = pd.read_csv("Data/B_Helix/electric_field_G_CO_head.dat", header=None, sep='\s+')

X_CCC_body_B = X_CCC_body_B.iloc[:, [0, 1, 2, 6, 7, 8]]
X_CCC_head_B = X_CCC_head_B.iloc[:, [0, 1, 2, 6, 7, 8]]
X_CCC_body_B = X_CCC_body_B.values
X_CCC_head_B = X_CCC_head_B.values
X_CCO_body_B = X_CCO_body_B.values
X_CCO_head_B = X_CCO_head_B.values
X_GCO_body_B = X_GCO_body_B.values
X_GCO_head_B = X_GCO_head_B.values

# Read Dataset A
true_results_A = pd.read_csv("A-helix.csv", header=None)
true_results_A = true_results_A.values
true_freq_A = torch.FloatTensor(true_results_A[:, 0])
true_inten_A = torch.FloatTensor(true_results_A[:, 1])

X_CCC_body_A = pd.read_csv("Data/A_Helix/electric_field_C_CC_body.dat", header=None, sep='\s+')
X_CCC_head_A = pd.read_csv("Data/A_Helix/electric_field_C_CC_head.dat", header=None, sep='\s+')
X_CCO_body_A = pd.read_csv("Data/A_Helix/electric_field_C_CO_body.dat", header=None, sep='\s+')
X_CCO_head_A = pd.read_csv("Data/A_Helix/electric_field_C_CO_head.dat", header=None, sep='\s+')
X_GCO_body_A = pd.read_csv("Data/A_Helix/electric_field_G_CO_body.dat", header=None, sep='\s+')
X_GCO_head_A = pd.read_csv("Data/A_Helix/electric_field_G_CO_head.dat", header=None, sep='\s+')

X_CCC_body_A = X_CCC_body_A.iloc[:, [0, 1, 2, 6, 7, 8]]
X_CCC_head_A = X_CCC_head_A.iloc[:, [0, 1, 2, 6, 7, 8]]
X_CCC_body_A = X_CCC_body_A.values
X_CCC_head_A = X_CCC_head_A.values
X_CCO_body_A = X_CCO_body_A.values
X_CCO_head_A = X_CCO_head_A.values
X_GCO_body_A = X_GCO_body_A.values
X_GCO_head_A = X_GCO_head_A.values

X_CCC_head_standardizer = StandardScaler()
X_CCO_head_standardizer = StandardScaler()
X_GCO_head_standardizer = StandardScaler()
X_CCC_body_standardizer = StandardScaler()
X_CCO_body_standardizer = StandardScaler()
X_GCO_body_standardizer = StandardScaler()
X_CCC_head_standardizer.fit(np.vstack((X_CCC_head_B, X_CCC_head_A)))
X_CCO_head_standardizer.fit(np.vstack((X_CCO_head_B, X_CCO_head_A)))
X_GCO_head_standardizer.fit(np.vstack((X_GCO_head_B, X_GCO_head_A)))
X_CCC_body_standardizer.fit(np.vstack((X_CCC_body_B, X_CCC_body_A)))
X_CCO_body_standardizer.fit(np.vstack((X_CCO_body_B, X_CCO_body_A)))
X_GCO_body_standardizer.fit(np.vstack((X_GCO_body_B, X_GCO_body_A)))

X_CCC_head_B = X_CCC_head_standardizer.transform(X_CCC_head_B)
X_CCO_head_B = X_CCO_head_standardizer.transform(X_CCO_head_B)
X_GCO_head_B = X_GCO_head_standardizer.transform(X_GCO_head_B)
X_CCC_head_B = torch.FloatTensor(X_CCC_head_B)
X_CCO_head_B = torch.FloatTensor(X_CCO_head_B)
X_GCO_head_B = torch.FloatTensor(X_GCO_head_B)

X_CCC_body_B = X_CCC_body_standardizer.transform(X_CCC_body_B)
X_CCO_body_B = X_CCO_body_standardizer.transform(X_CCO_body_B)
X_GCO_body_B = X_GCO_body_standardizer.transform(X_GCO_body_B)
X_CCC_body_B = torch.FloatTensor(X_CCC_body_B)
X_CCO_body_B = torch.FloatTensor(X_CCO_body_B)
X_GCO_body_B = torch.FloatTensor(X_GCO_body_B)

X_CCC_head_A = X_CCC_head_standardizer.transform(X_CCC_head_A)
X_CCO_head_A = X_CCO_head_standardizer.transform(X_CCO_head_A)
X_GCO_head_A = X_GCO_head_standardizer.transform(X_GCO_head_A)
X_CCC_body_A = X_CCC_body_standardizer.transform(X_CCC_body_A)
X_CCO_body_A = X_CCO_body_standardizer.transform(X_CCO_body_A)
X_GCO_body_A = X_GCO_body_standardizer.transform(X_GCO_body_A)

X_CCC_head_A = torch.FloatTensor(X_CCC_head_A)
X_CCO_head_A = torch.FloatTensor(X_CCO_head_A)
X_GCO_head_A = torch.FloatTensor(X_GCO_head_A)
X_CCC_body_A = torch.FloatTensor(X_CCC_body_A)
X_CCO_body_A = torch.FloatTensor(X_CCO_body_A)
X_GCO_body_A = torch.FloatTensor(X_GCO_body_A)

# set up CPU/GPU
device = torch.device("cpu")
# device = torch.device("cuda:0")
dim_hidden = dim_model
layer_num = layer_num
dropout = 0.1
dim_reg = 64

folder = "./Train/layer_" + str(layer_num) + " dim_" + str(dim_model) + " lr_" + str(lr) + " wd_" + str(wd) + "/"
Path(folder).mkdir(parents=True, exist_ok=True)

# Index for B
N_B = 9600000
all_magnitude_B = torch.zeros(N_B).to(device)
G_CO_index_B = np.arange(0, N_B, 3)
G_CO_index_B = np.sort(np.hstack(G_CO_index_B))
C_CO_index_B = np.arange(1, N_B, 3)
C_CO_index_B = np.sort(np.hstack(C_CO_index_B))
C_CC_index_B = np.arange(2, N_B, 3)
C_CC_index_B = np.sort(np.hstack(C_CC_index_B))

# Index for A
N_A = 6000000
all_magnitude = torch.zeros(N_A).to(device)
G_CO_index_A = []
C_CO_index_A = []
C_CC_index_A = []
for i in range(5):
    G_CO_index_A.append(np.arange(i, N_A, 30))
    G_CO_index_A.append(np.arange(i+15, N_A, 30))
G_CO_index_A = np.sort(np.hstack(G_CO_index_A))

for i in range(5, 15, 2):
    C_CO_index_A.append(np.arange(i, N_A, 30))
    C_CO_index_A.append(np.arange(i + 15, N_A, 30))
    C_CC_index_A.append(np.arange(i+1, N_A, 30))
    C_CC_index_A.append(np.arange(i + 16, N_A, 30))
C_CO_index_A = np.sort(np.hstack(C_CO_index_A))
C_CC_index_A = np.sort(np.hstack(C_CC_index_A))

try:
    ck = folder + "best_checkpoint.pth"
    model, optimizer, _ = load_model(ck, device)
except:
    # instantiate an optimizer
    model = MagnitudeModel([6, 9, 9], dim_hidden, layer_num, dropout, dim_reg)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
# send model to CPU/GPU
_ = model.to(device)

epoch = 500
loss_train_history = []
loss_valid_history = []
# use MSE loss
min_valid_loss = sys.maxsize
best_epoch = 0
early_stop_indicator = 0
for k in range(epoch):
    epoch_start = time.time()
    if k and k % save_model_every == 0:
        save_model(folder, k, [6, 9, 9], dim_hidden, layer_num, dropout, dim_reg, model, optimizer)
    # set model training state
    model.train()
    forward_calculate_start = time.time()
    C_CC_magnitude_head_B, C_CC_magnitude_body_B, C_CO_magnitude_head_B, C_CO_magnitude_body_B, G_CO_magnitude_head_B, \
    G_CO_magnitude_body_B = model.forward(X_CCC_head_B, X_CCC_body_B, X_CCO_head_B, X_CCO_body_B, X_GCO_head_B,
                                          X_GCO_body_B)

    G_CO_magnitude_B = G_combine_B(G_CO_magnitude_head_B, G_CO_magnitude_body_B)
    C_CO_magnitude_B = C_combine_B(C_CO_magnitude_head_B, C_CO_magnitude_body_B)
    C_CC_magnitude_B = C_combine_B(C_CC_magnitude_head_B, C_CC_magnitude_body_B)
    all_magnitude_B = torch.zeros(N_B).to(device)
    all_magnitude_B[G_CO_index_B, ] = G_CO_magnitude_B.reshape(-1,)
    all_magnitude_B[C_CO_index_B, ] = C_CO_magnitude_B.reshape(-1,)
    all_magnitude_B[C_CC_index_B, ] = C_CC_magnitude_B.reshape(-1,)
    all_magnitude_B = all_magnitude_B.reshape(-1, 48, 1)
    inten_B, freq_B = spB.simply(all_magnitude_B)

    all_magnitude_A = torch.zeros(N_A).to(device)
    C_CC_magnitude_head_A, C_CC_magnitude_body_A, C_CO_magnitude_head_A, C_CO_magnitude_body_A, G_CO_magnitude_head_A, \
    G_CO_magnitude_body_A = model.forward(X_CCC_head_A, X_CCC_body_A, X_CCO_head_A, X_CCO_body_A, X_GCO_head_A,
                                          X_GCO_body_A)
    G_CO_magnitude_A = G_combine_A(G_CO_magnitude_head_A, G_CO_magnitude_body_A)
    C_CO_magnitude_A = C_combine_A(C_CO_magnitude_head_A, C_CO_magnitude_body_A)
    C_CC_magnitude_A = C_combine_A(C_CC_magnitude_head_A, C_CC_magnitude_body_A)
    all_magnitude_A[G_CO_index_A, ] = G_CO_magnitude_A.reshape(-1, )
    all_magnitude_A[C_CO_index_A, ] = C_CO_magnitude_A.reshape(-1, )
    all_magnitude_A[C_CC_index_A, ] = C_CC_magnitude_A.reshape(-1, )
    all_magnitude_A = all_magnitude_A.reshape(-1, 30, 1)
    inten_A, freq_A = spA.simply(all_magnitude_A)
    forward_calculate_end = time.time()
    forward_time = forward_calculate_end - forward_calculate_start
    print("Forward calculation time cost is ", forward_time)

    loss_calculate_start = time.time()
    # loss = curve_loss((freq, inten), (true_freq, true_inten))
    loss_B = local_peak_loss((freq_B, inten_B), (true_freq_B, true_inten_B), "B")
    loss_A = local_peak_loss((freq_A, inten_A), (true_freq_A, true_inten_A), "A")
    loss = loss_A + loss_B
    loss_calculate_end = time.time()
    if k == 0:
        min_valid_loss = loss
    loss_time = loss_calculate_end - loss_calculate_start
    print("Loss calculation time cost is ", loss_time)

    optimization_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimization_end = time.time()
    optimization_time = optimization_end - optimization_start
    print("Optimization time cost is ", optimization_time)

    model.eval()
    # Evaluate B
    all_magnitude_B = torch.zeros(N_B).to(device)
    C_CC_magnitude_head_B, C_CC_magnitude_body_B, C_CO_magnitude_head_B, C_CO_magnitude_body_B, G_CO_magnitude_head_B, \
    G_CO_magnitude_body_B = model.forward(X_CCC_head_B, X_CCC_body_B, X_CCO_head_B, X_CCO_body_B, X_GCO_head_B,
                                          X_GCO_body_B)
    G_CO_magnitude_B = G_combine_B(G_CO_magnitude_head_B, G_CO_magnitude_body_B)
    C_CO_magnitude_B = C_combine_B(C_CO_magnitude_head_B, C_CO_magnitude_body_B)
    C_CC_magnitude_B = C_combine_B(C_CC_magnitude_head_B, C_CC_magnitude_body_B)
    all_magnitude_B[G_CO_index_B, ] = G_CO_magnitude_B.reshape(-1, )
    all_magnitude_B[C_CO_index_B, ] = C_CO_magnitude_B.reshape(-1, )
    all_magnitude_B[C_CC_index_B, ] = C_CC_magnitude_B.reshape(-1, )
    all_magnitude_B = all_magnitude_B.reshape(-1, 48, 1)
    inten_B, freq_B = spB.simply(all_magnitude_B)
    loss_B = local_peak_loss((freq_B, inten_B), None, "B")

    # Evaluate A
    all_magnitude_A = torch.zeros(N_A).to(device)
    C_CC_magnitude_head_A, C_CC_magnitude_body_A, C_CO_magnitude_head_A, C_CO_magnitude_body_A, G_CO_magnitude_head_A, \
    G_CO_magnitude_body_A = model.forward(X_CCC_head_A, X_CCC_body_A, X_CCO_head_A, X_CCO_body_A, X_GCO_head_A,
                                          X_GCO_body_A)
    G_CO_magnitude_A = G_combine_A(G_CO_magnitude_head_A, G_CO_magnitude_body_A)
    C_CO_magnitude_A = C_combine_A(C_CO_magnitude_head_A, C_CO_magnitude_body_A)
    C_CC_magnitude_A = C_combine_A(C_CC_magnitude_head_A, C_CC_magnitude_body_A)
    all_magnitude_A[G_CO_index_A, ] = G_CO_magnitude_A.reshape(-1, )
    all_magnitude_A[C_CO_index_A, ] = C_CO_magnitude_A.reshape(-1, )
    all_magnitude_A[C_CC_index_A, ] = C_CC_magnitude_A.reshape(-1, )
    all_magnitude_A = all_magnitude_A.reshape(-1, 30, 1)
    inten_A, freq_A = spA.simply(all_magnitude_A)
    loss_A = local_peak_loss((freq_A, inten_A), None, "A")

    loss_train_history.append(loss.item())
    loss_valid_history.append(loss_A.item())

    loss = loss_A + loss_B
    if loss < min_valid_loss:
        min_valid_loss = loss
        early_stop_indicator = 0
        best_epoch = k
        save_model(folder, k, [6, 9, 9], dim_hidden, layer_num, dropout, dim_reg, model, optimizer, best=True)
    else:
        early_stop_indicator += 1
    if early_stop_indicator > 200:
        break
    epoch_end = time.time()

    print("epoch:", k + 1, ": , loss = ", np.around(loss.item(), 5), "loss B = ", np.around(loss_B.item(), 5),
          ", loss A = ", np.around(loss_A.item(), 5),
          "; Epoch time cost:", epoch_end-epoch_start)

print(best_epoch)

# Best B
C_CC_magnitude_head_B, C_CC_magnitude_body_B, C_CO_magnitude_head_B, C_CO_magnitude_body_B, G_CO_magnitude_head_B, \
G_CO_magnitude_body_B, best_epoch_B = make_prediction(X_CCC_head_B, X_CCC_body_B, X_CCO_head_B, X_CCO_body_B,
                                                      X_GCO_head_B, X_GCO_body_B, folder=folder)
G_CO_magnitude_B = G_combine_B(G_CO_magnitude_head_B, G_CO_magnitude_body_B)
C_CO_magnitude_B = C_combine_B(C_CO_magnitude_head_B, C_CO_magnitude_body_B)
C_CC_magnitude_B = C_combine_B(C_CC_magnitude_head_B, C_CC_magnitude_body_B)

all_magnitude_B = torch.zeros(N_B).to(device)
all_magnitude_B[G_CO_index_B, ] = G_CO_magnitude_B.reshape(-1,)
all_magnitude_B[C_CO_index_B, ] = C_CO_magnitude_B.reshape(-1,)
all_magnitude_B[C_CC_index_B, ] = C_CC_magnitude_B.reshape(-1,)
all_magnitude_B = all_magnitude_B.reshape(-1, 48, 1)
inten_B, freq_B = spB.simply(all_magnitude_B)
loss_B = local_peak_loss((freq_B, inten_B), (true_freq_B, true_inten_B), "B")
inten_B = inten_B.detach().cpu().numpy()
freq_B = freq_B.detach().cpu().numpy()
plot_data_B = np.hstack((freq_B.reshape(-1, 1), inten_B.reshape(-1, 1)))
plot_data_df_B = pd.DataFrame(plot_data_B)
all_magnitude = all_magnitude.reshape(-1, 1)
all_magnitude = all_magnitude.detach().cpu().numpy()
all_magnitude_df_B = pd.DataFrame(all_magnitude)

# Best A
C_CC_magnitude_head_A, C_CC_magnitude_body_A, C_CO_magnitude_head_A, C_CO_magnitude_body_A, G_CO_magnitude_head_A, \
G_CO_magnitude_body_A, _ = make_prediction(X_CCC_head_A, X_CCC_body_A, X_CCO_head_A, X_CCO_body_A,
                                           X_GCO_head_A, X_GCO_body_A, folder=folder)
G_CO_magnitude_A = G_combine_A(G_CO_magnitude_head_A, G_CO_magnitude_body_A)
C_CO_magnitude_A = C_combine_A(C_CO_magnitude_head_A, C_CO_magnitude_body_A)
C_CC_magnitude_A = C_combine_A(C_CC_magnitude_head_A, C_CC_magnitude_body_A)

all_magnitude_A = torch.zeros(N_A).to(device)
all_magnitude_A[G_CO_index_A, ] = G_CO_magnitude_A.reshape(-1, )
all_magnitude_A[C_CO_index_A, ] = C_CO_magnitude_A.reshape(-1, )
all_magnitude_A[C_CC_index_A, ] = C_CC_magnitude_A.reshape(-1, )
all_magnitude_A = all_magnitude_A.reshape(-1, 30, 1)
inten_A, freq_A = spA.simply(all_magnitude_A)
loss_A = local_peak_loss((freq_A, inten_A), (true_freq_A, true_inten_A), "A")
inten_A = inten_A.detach().cpu().numpy()
freq_A = freq_A.detach().cpu().numpy()
plot_data_A = np.hstack((freq_A.reshape(-1, 1), inten_A.reshape(-1, 1)))
plot_data_df_A = pd.DataFrame(plot_data_A)
all_magnitude_A = all_magnitude_A.reshape(-1, 1)
all_magnitude_A = all_magnitude_A.detach().cpu().numpy()
all_magnitude_df_A = pd.DataFrame(all_magnitude_A)

results = "./Prediction/layer_" + str(layer_num) + " dim_" + str(dim_model) + " lr_" + str(lr) + " wd_" + str(wd) + "/"
Path(results).mkdir(parents=True, exist_ok=True)
plot_data_df_A.to_csv(results + "test_A_plot_data.csv.csv", index=False, header=False)
all_magnitude_df_A.to_csv(results + "A_magnitude_pred.csv", index=False,  header=False)
plot_data_df_B.to_csv(results + "test_B_plot_data.csv.csv", index=False, header=False)
all_magnitude_df_B.to_csv(results + "B_magnitude_pred.csv", index=False,  header=False)

output_results = [layer_num, dim_model, lr, wd, loss_B.item(), loss_A.item()]
file = open("Current Best Results.csv", mode="a")
writer = csv.writer(file)
writer.writerow(output_results)
file.close()