import torch
from Models import *
from torch.optim import Adam


def save_model(folder, k, input_dim, dim_hidden, layer_num, dropout, dim_reg, model, optimizer, best=False):
    checkpoint = {'input_dim': input_dim,
                  'dim_hidden': dim_hidden,
                  'layer_num': layer_num,
                  'dropout': dropout,
                  'dim_reg': dim_reg,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': k}
    if best:
        k = "best"
    torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


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


def G_combine_B(head_mag, body_mag, N=3200000):
    mag = torch.zeros(N, head_mag.shape[1]).to(head_mag.device)
    head_index = np.arange(0, N, 8)
    body_index = np.arange(0, N)
    body_index = np.delete(body_index, head_index)
    mag[head_index, ] = head_mag
    mag[body_index, ] = body_mag
    return mag


def C_combine_B(head_mag, body_mag, N=3200000):
    mag = torch.zeros(N, head_mag.shape[1]).to(head_mag.device)
    head_index = np.arange(7, N, 8)
    body_index = np.arange(0, N)
    body_index = np.delete(body_index, head_index)
    mag[head_index, ] = head_mag
    mag[body_index, ] = body_mag
    return mag


def G_combine_A(head_mag, body_mag, N=2000000):
    mag = torch.zeros(N, head_mag.shape[1]).to(head_mag.device)
    head_index = np.arange(0, N, 5)
    body_index = np.arange(0, N)
    body_index = np.delete(body_index, head_index)
    mag[head_index, ] = head_mag
    mag[body_index, ] = body_mag
    return mag


def C_combine_A(head_mag, body_mag, N=2000000):
    mag = torch.zeros(N, head_mag.shape[1]).to(head_mag.device)
    head_index = np.arange(4, N, 5)
    body_index = np.arange(0, N)
    body_index = np.delete(body_index, head_index)
    mag[head_index, ] = head_mag
    mag[body_index, ] = body_mag
    return mag


def local_peak_loss(pred, true, input_type="B", ratio=0.5):
    compute_loss = nn.MSELoss()
    if input_type == "A":
        peak_true = torch.FloatTensor([0.1532, 0.86515, 1])
        valley_true = torch.FloatTensor([0.6724])
        width_true = torch.FloatTensor([53.39509999999996])
        peak_pos = torch.FloatTensor([1617.69465, 1650.84327, 1681.53233])
        valley_pos = torch.FloatTensor([1664.71929])
    elif input_type == "B":
        peak_true = torch.FloatTensor([0.10881, 0.68999, 0.69411, 1])
        valley_true = torch.FloatTensor([0.55722])
        width_true = torch.FloatTensor([51.65726000000018])
        peak_pos = torch.FloatTensor([1621.49026, 1650.52027, 1656.73477, 1683.69753])
        valley_pos = torch.FloatTensor([1668.39139])
    else:
        raise ValueError("Type must be chosen from ['A', 'B']")

    freq, inten = pred
    freq_true, inten_true = true

    # Define the consider range
    consider_range = torch.logical_and(freq >= 1580, freq <= 1720)[1:-1]

    # Find the predicted peaks
    previous_smaller_index = inten[1:-1] - inten[0:-2] > 0
    next_smaller_index = inten[1:-1] - inten[2:] > 0
    peak_index = torch.logical_and(previous_smaller_index, next_smaller_index)
    peak_index = torch.logical_and(peak_index, consider_range)
    peak_index = torch.where(peak_index)[0] + 1
    peak_pred = inten[peak_index]
    peak_pos_pred = freq[peak_index]
    # reverse peak order
    peak_pred = torch.flip(peak_pred, dims=(0,))
    peak_true = torch.flip(peak_true, dims=(0,))
    peak_pos = torch.flip(peak_pos, dims=(0,))
    peak_pos_pred = torch.flip(peak_pos_pred, dims=(0,))

    if input_type == "A":
        if peak_pred.shape[0] > 3:
            peak_true = torch.hstack((peak_true, torch.zeros(peak_pred.shape[0] - 3)))
            peak_pos = torch.hstack((peak_pos, torch.zeros(peak_pos_pred.shape[0] - 3)))
        else:
            peak_true = peak_true[0:peak_pred.shape[0]]
            peak_pos = peak_pos[0:peak_pos_pred.shape[0]]
    if input_type == "B":
        if peak_pred.shape[0] > 4:
            peak_true = torch.hstack((peak_true, torch.zeros(peak_pred.shape[0] - 4)))
            peak_pos = torch.hstack((peak_pos, torch.zeros(peak_pos_pred.shape[0] - 4)))
        else:
            peak_true = peak_true[0:peak_pred.shape[0]]
            peak_pos = peak_pos[0:peak_pos_pred.shape[0]]

    # Local peak loss
    local_loss = 0
    for i in range(peak_pos.shape[0]):
        peak_x_tmp = freq_true - peak_pos[i]
        peak_x_pred_tmp = freq - peak_pos_pred[i]
        local_index_true = torch.abs(peak_x_tmp) < 5
        local_index_pred = torch.abs(peak_x_pred_tmp) < 5

        freq_loc = freq[local_index_pred]
        inten_loc = inten[local_index_pred]
        freq_loc_true = freq_true[local_index_true]
        inten_loc_true = inten_true[local_index_true]

        inten_pred_loc = torch.zeros(inten_loc_true.shape)
        for j in range(freq_loc_true.shape[0]):
            x = freq_loc_true[j]
            index = torch.sort(torch.abs(x - freq_loc)).indices[0:2]
            freq_tmp = freq_loc[index]
            inten_tmp = inten_loc[index]
            y = (inten_tmp[1] - inten_tmp[0]) / (freq_tmp[1] - freq_tmp[0]) * (x - freq_tmp[0]) + inten_tmp[0]
            inten_pred_loc[j] = y
        local_loss += compute_loss(inten_loc_true, inten_pred_loc)

    # Find the predicted valleys
    previous_larger_index = inten[1:-1] - inten[0:-2] < 0
    next_larger_index = inten[1:-1] - inten[2:] < 0
    valley_index = torch.logical_and(previous_larger_index, next_larger_index)
    valley_index = torch.logical_and(valley_index, consider_range)
    valley_index = torch.where(valley_index)[0] + 1
    valley_pred = inten[valley_index]
    valley_pos_pred = freq[valley_index]

    # reverse valley order
    valley_pred = torch.flip(valley_pred, dims=(0,))
    valley_pos_pred = torch.flip(valley_pos_pred, dims=(0,))
    if input_type == "A":
        if valley_pred.shape[0] < 1:
            valley_pred = torch.FloatTensor([1])
            valley_pos_pred = torch.FloatTensor([1])
        else:
            valley_pred = valley_pred[0]  # only consider the 1st valley
            valley_pos_pred = valley_pos_pred[0]
    if input_type == "B":
        if valley_pred.shape[0] < 1:
            valley_pred = torch.FloatTensor([1])
            valley_pos_pred = torch.FloatTensor([1])
        else:
            valley_pred = valley_pred[0]  # only consider the 1st valley
            valley_pos_pred = valley_pos_pred[0]

    # Calculate predict width
    pred_width_index = torch.where(torch.abs(inten - 0.5) < 0.005)[0]
    if pred_width_index.shape[0] < 2:
        width_pred = torch.FloatTensor([1000])
    else:
        width_left = freq[pred_width_index[0]]
        width_right = freq[pred_width_index[-1]]
        width_pred = width_right - width_left

    if input_type == "A":
        if peak_pred.shape[0] >= 3:
            peak_pred[0:2] = peak_pred[0:2]
            peak_true[0:2] = peak_true[0:2]
            peak_pred[2:] = peak_pred[2:] / 2
            peak_true[2:] = peak_true[2:] / 2
        else:
            peak_pred = ratio * peak_pred
            peak_true = ratio * peak_true
    if input_type == "B":
        if peak_pred.shape[0] >= 4:
            peak_pred[0:3] = peak_pred[0:3]
            peak_true[0:3] = peak_true[0:3]
            peak_pred[3:] = peak_pred[3:] / 2
            peak_true[3:] = peak_true[3:] / 2
        else:
            peak_pred = peak_pred
            peak_true = peak_true
    pred_char = torch.hstack((peak_pred, valley_pred, ratio*width_pred, ratio*valley_pos_pred, ratio*peak_pos_pred))
    true_char = torch.hstack((peak_true, valley_true, ratio*width_true, ratio*valley_pos, ratio*peak_pos))
    loss = compute_loss(true_char, pred_char)
    if input_type == "B":
        if peak_pred.shape[0] >= 3:
            loss = loss + torch.square(peak_pred[1] - peak_pred[2])
    loss = loss + local_loss
    return loss