import math
import torch
import numpy as np
import pickle as pkl
import time

nchrom = 30
nsnpsht = 200000
tstep = 300
f_tstep = 8000
interval = 300
c = 0.0299792458
PI = math.acos(-1)
conve = 2*PI*c
dt = 0.01
dw = 2.0*PI/dt/f_tstep
minfreq = 1500.00
maxfreq = 1800.00
im = (0.0, 1.0)
T1 = 0.649*2.0

start_time = time.time()
# Load the array with Pickle
with open('./Data/Python_Data/A_Helix/F_real.pkl', 'rb') as f_real:
    F_real_all = torch.FloatTensor(pkl.load(f_real))
with open('./Data/Python_Data/A_Helix/F_imag.pkl', 'rb') as f_imag:
    F_imag_all = torch.FloatTensor(pkl.load(f_imag))
with open('./Data/Python_Data/A_Helix/unit_td.pkl', 'rb') as f_unit_td:
    unit_td = torch.FloatTensor(pkl.load(f_unit_td))
end_time = time.time()
print("Loading Data Cost Time:", end_time-start_time)

dintensity = torch.FloatTensor(np.zeros(f_tstep+1))
actualw = (np.arange(f_tstep)+1)*dw
index_bool = np.logical_and((minfreq*conve) <= actualw, actualw <= (maxfreq*conve))
num_feasible = np.sum(index_bool)

actualw = torch.FloatTensor(actualw)
actualw_feasible = actualw[index_bool].reshape(-1, 1)
time_list = torch.FloatTensor(np.arange(f_tstep+1)*dt).reshape(1, -1)
exp_constant = torch.exp(-torch.vstack([time_list]*num_feasible)/T1)
b_tmp = np.matmul(actualw_feasible, time_list)
b_real = torch.FloatTensor(np.cos(b_tmp))
b_imag = torch.FloatTensor(-np.sin(b_tmp))


def simply(magnitude):
    f_count = 0
    m = unit_td*magnitude
    mFm_real = torch.FloatTensor(np.zeros(f_tstep+1))
    mFm_imag = torch.FloatTensor(np.zeros(f_tstep+1))
    # calculate <mFm> real and imag part separate
    for k in range(tstep+1):
        index_a = np.arange(0, nsnpsht-k, interval)
        index_b = index_a + k
        m_a_tmp = m[index_a, ]
        m_b_tmp = m[index_b, ]
        
        F_real = F_real_all[f_count:(f_count+len(index_a)), ]
        F_imag = F_imag_all[f_count:(f_count+len(index_a)), ]
        f_count = f_count + len(index_a)
        
        mFm_real_tmp = torch.bmm(torch.bmm(torch.transpose(m_a_tmp, 2, 1), F_real), m_b_tmp)
        mFm_real_tmp = mFm_real_tmp.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        mFm_imag_tmp = torch.bmm(torch.bmm(torch.transpose(m_a_tmp, 2, 1), F_imag), m_b_tmp)
        mFm_imag_tmp = mFm_imag_tmp.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        mFm_real[k] = torch.mean(mFm_real_tmp)/3
        mFm_imag[k] = torch.mean(mFm_imag_tmp)/3
    
    inten = torch.FloatTensor(np.zeros(f_tstep))
    freq = torch.FloatTensor(np.zeros(f_tstep))
    dintensity = (b_real*torch.vstack([mFm_real]*num_feasible) - b_imag*torch.vstack([mFm_imag]*num_feasible)) * exp_constant
    inten[index_bool] = torch.sum(dintensity[:, 0:-1] + dintensity[:, 1:], 1)*dt/2
    freq[index_bool] = actualw_feasible.reshape(-1)/conve
    inten = inten/torch.max(inten)
    return inten, freq

