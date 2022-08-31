import numpy as np
import pandas as pd
import random
from osgeo import gdal_array
from datetime import datetime
from os import path
from skimage import io

import torch
import utils

from torchvision import models
from model import MSMT


from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasetName = 'DFC2018'
#datasetName = 'Vaihingen'

if datasetName=='DFC2018':
    label_codes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

if datasetName == 'Vaihingen':
    label_codes = [(255,255,255), (0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0)]

id2code = {k: v for k, v in enumerate(label_codes)}

decay = False
save = True

lr = 0.0001
batchSize = 1
numEpochs = 2
training_samples = 1000
# val_freq = 1000
train_iters = int(training_samples/batchSize)
cropSize = 256


all_rgb, _, all_sem, all_hsi = collect_tilenames("train", datasetName)
# todo: tile the sem for val
# val_rgb, val_dsm, val_sem = collect_tilenames("val", datasetName)

NUM_TRAIN_IMAGES = len(all_rgb)
# NUM_VAL_IMAGES = len(val_rgb)


net = MSMT(device, input_dim_rgb=3, input_dim_hsi=50, nz=8).to(device)
log_var_ls = [torch.nn.Parameter(torch.zeros((1,), requires_grad=True, device=device)) for _ in range(6)]
std_ls = [torch.exp(log_var_i)**0.5 for log_var_i in log_var_ls]
print(std_ls)

MSE = torch.nn.MSELoss()
CCE = torch.nn.CrossEntropyLoss()


# params = ([p for p in net.parameters()] + log_var_ls)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=train_iters, epochs=numEpochs)


min_loss = 1000

for current_epoch in range(1, numEpochs+1):
    print(f"Current epoch:{current_epoch}")

    error_AL = []
    error_L_auto_rgb = []
    error_L_trans_rgb = []
    error_L_auto_hsi = []
    error_L_trans_hsi = []
    error_L_down = []
    error_L_content = []
    ep_lr = []

    for iters in range(train_iters):
        idx = random.randint(0, len(all_rgb)-1)

        rgb_batch = []
        sem_batch = []
        hsi_batch = []

        rgb_tile = np.array(Image.open(all_rgb[idx])) / 255
        sem_tile = np.array(Image.open(all_sem[idx]))
        hsi_tile = gdal_array.LoadFile(all_hsi[idx], buf_xsize=1192, buf_ysize=1202)

        for i in range(hsi_tile.shape[0]):
            temp_max = hsi_tile[i].max()
            temp_min = hsi_tile[i].min()
            hsi_tile[i] = (hsi_tile[i] - temp_min) / (temp_max - temp_min)
        hsi_tile = hsi_tile.transpose([1, 2, 0])
        hsi_tile = hsi_tile.astype(np.float32)

        for i in range(batchSize):
            h = hsi_tile.shape[0]
            w = hsi_tile.shape[1]
            r = random.randint(0, h - cropSize)
            c = random.randint(0, w - cropSize)
            rgb = rgb_tile[r:r + cropSize, c:c + cropSize]
            sem = sem_tile[r:r + cropSize, c:c + cropSize]
            hsi = hsi_tile[r:r+cropSize, c:c+cropSize]
            # if datasetName == 'DFC2018':
            #     sem = sem[..., np.newaxis]

            rgb_batch.append(rgb)
            hsi_batch.append(hsi)
            # sem_batch.append(rgb_to_onehot(sem, datasetName, id2code))
            sem_batch.append(sem)

        rgb_batch = np.array(rgb_batch).transpose([0, 3, 1, 2])
        sem_batch = np.array(sem_batch)        #.transpose([0, 3, 1, 2])
        hsi_batch = np.array(hsi_batch).transpose([0, 3, 1, 2])

        rgb_batch = torch.tensor(rgb_batch, dtype=torch.float32, device=device)
        sem_batch = torch.tensor(sem_batch, dtype=torch.long, device=device)
        hsi_batch = torch.tensor(hsi_batch, dtype=torch.float32, device=device)

        # with torch.autograd.set_detect_anomaly(True):
        current_lr = scheduler.get_last_lr()
        optimizer.zero_grad()

        row_rgb, row_hsi, rgb_content, hsi_content, gen_rgb, gen_hsi, transfer_gen_rgb, transfer_gen_hsi, down_out = net(rgb_batch, hsi_batch)

        L_auto_rgb = MSE(row_rgb, gen_rgb)
        L_auto_hsi = MSE(row_hsi, gen_hsi)
        # todo: figure out what trans-loss is
        L_trans_rgb = MSE(row_rgb, transfer_gen_rgb)
        L_trans_hsi = MSE(row_hsi, transfer_gen_hsi)
        L_down = CCE(down_out, sem_batch)
        L_content = MSE(rgb_content, hsi_content)

        loss_ls = [L_auto_rgb, L_auto_hsi, L_trans_rgb, L_trans_hsi, L_down, L_content]
        temp_loss = 0
        for loss_i, weight_i in zip(loss_ls, log_var_ls):
            temp_loss = temp_loss + torch.exp(-weight_i) * loss_i + weight_i

        total_loss = temp_loss

        # print(total_loss.item())
        print(L_down.item())

        # total_loss.backward()
        L_down.backward()
        optimizer.step()
        scheduler.step()

        error_AL.append(total_loss.item())
        error_L_auto_rgb.append(L_auto_rgb.item())
        error_L_trans_rgb.append(L_trans_rgb.item())
        error_L_auto_hsi.append(L_auto_hsi.item())
        error_L_trans_hsi.append(L_trans_hsi.item())
        error_L_down.append(L_down.item())
        error_L_content.append(L_content.item())
        ep_lr.append(current_lr)

        # every train_iters//5 save the output for visualization
        if iters in list(range(0, train_iters, 100)):
        #     # todo: save result and visualization
        #     epoch_result([row_rgb.cpu().numpy(), row_hsi.cpu().numpy(), rgb_content.detach().cpu().numpy(), hsi_content.detach().cpu().numpy(),
        #                          gen_rgb.detach().cpu().numpy(), gen_hsi.detach().cpu().numpy(), transfer_gen_rgb.detach().cpu().numpy(), transfer_gen_hsi.detach().cpu().numpy(), down_out.detach().cpu().numpy()])

            # down_out = torch.stack([onehot_to_rgb(down_out[i]).unsqueeze(0) for i in range(down_out.shape[0])], dim=0)
            # sem_batch = torch.stack([onehot_to_rgb(sem_batch[i]).unsqueeze(0) for i in range(sem_batch.shape[0])], dim=0)
            down_out = down_out.argmax(dim=1).float()
            sem_batch = sem_batch.float()
            for img, filename in zip([row_rgb, gen_rgb, transfer_gen_rgb, down_out, sem_batch], [f'./results/train_img_down_only/epoch_{current_epoch}_iters_{iters}_row_rgb.jpg', f'./results/train_img_down_only/epoch_{current_epoch}_iters_{iters}_gen_rgb.jpg',
                                                                                                 f'./results/train_img_down_only/epoch_{current_epoch}_iters_{iters}_transfer_gen_rgb.jpg', f'./results/train_img_down_only/epoch_{current_epoch}_iters_{iters}_down_out.jpg',
                                                                                                 f'./results/train_img_down_only/epoch_{current_epoch}_iters_{iters}_sem_batch.jpg']):
                torchvision.utils.save_image(img, filename, nrow=2)

    epoch_loss = np.array([error_AL, error_L_auto_rgb, error_L_trans_rgb, error_L_auto_hsi, error_L_trans_hsi, error_L_down, error_L_content, ep_lr]).transpose([1, 0])
    epoch_loss = pd.DataFrame(epoch_loss)
    epoch_loss.columns = ['error_AL', 'error_L_auto_rgb', 'error_L_trans_rgb', 'error_L_auto_hsi', 'error_L_trans_hsi', 'error_L_down', 'error_L_content', 'ep_lr']
    epoch_loss.to_csv(f'./results/loss_down_only/epoch{current_epoch}.csv', sep=' ')










