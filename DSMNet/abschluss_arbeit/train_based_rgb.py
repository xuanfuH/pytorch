import pandas as pd
import random
from model import MSMT
from torch.utils.data import Dataset, DataLoader

from utils import *

# set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


class DFC2018Dataset(Dataset):
    def __init__(self, train_or_val, transform=None):
        if train_or_val == 'train':
            self.data_num_list = np.arange(1, 1501)
            # self.data_num_list = np.arange(1, 10)
            self.n_samples = self.data_num_list.size
        elif train_or_val == 'valid':
            self.data_num_list = np.arange(1501, 2001)
            # self.data_num_list = np.arange(1501, 1510)
            self.n_samples = self.data_num_list.size
        else:
            raise ValueError('train_or_val must be defined as train or valid')
        self.transform = transform

    def __getitem__(self, item):
        rgb = np.array(Image.open('./train_data/rgb_{}.png'.format(self.data_num_list[item]))) / 255
        sem = np.array(Image.open('./train_data/sem_{}.png'.format(self.data_num_list[item])))
        hsi = np.load('./train_data/hsi_{}.npy'.format(self.data_num_list[item]))
        # dsm = np.load('./train_data/dsm_{}.npy'.format(self.data_num_list[item]))
        # norm = np.load('./train_data/norm_{}.npy'.format(self.data_num_list[item]))
        # sample = rgb, hsi, sem, dsm, norm
        sample = rgb, hsi, sem
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


lr = 0.0001
batchSize = 1
numEpochs = 2

train_dataset = DFC2018Dataset('train')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
valid_dataset = DFC2018Dataset('valid')
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batchSize)

net = MSMT(device, input_dim_rgb=3, input_dim_hsi=50, nz=8).to(device)
log_var_ls = [torch.nn.Parameter(torch.zeros((1,), requires_grad=True, device=device)) for _ in range(7)]
std_ls = [torch.exp(log_var_i)**0.5 for log_var_i in log_var_ls]
print(std_ls)

MSE = torch.nn.MSELoss()
CCE = torch.nn.CrossEntropyLoss()

params = ([p for p in net.parameters()] + log_var_ls)
optimizer = torch.optim.Adam(params, lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(len(train_dataset)/batchSize), epochs=numEpochs)
evaluator = Evaluator(21)

for current_epoch in range(1, numEpochs+1):
    error_AL = []
    error_L_auto_rgb = []
    error_L_trans_rgb = []
    error_L_auto_hsi = []
    error_L_trans_hsi = []
    error_L_down_rgb = []
    error_L_down_hsi = []
    error_L_content = []
    ep_lr = []

    net.train()
    evaluator.reset()
    for step, (rgb_batch, hsi_batch, sem_batch) in enumerate(train_dataloader, 0):
        rgb_batch = rgb_batch.type(torch.float32).permute([0, 3, 1, 2]).to(device)
        hsi_batch = hsi_batch.type(torch.float32).permute([0, 3, 1, 2]).to(device)
        sem_batch = sem_batch.type(torch.long).to(device)

        # with torch.autograd.set_detect_anomaly(True):
        current_lr = scheduler.get_last_lr()
        optimizer.zero_grad()

        row_rgb, row_hsi, rgb_content, hsi_content, gen_rgb, gen_hsi, transfer_gen_rgb, transfer_gen_hsi, down_out_rgb, down_out_hsi = net(rgb_batch, hsi_batch)

        L_auto_rgb = MSE(row_rgb, gen_rgb)
        L_auto_hsi = MSE(row_hsi, gen_hsi)
        # todo: figure out what trans-loss is
        L_trans_rgb = MSE(row_rgb, transfer_gen_rgb)
        L_trans_hsi = MSE(row_hsi, transfer_gen_hsi)
        L_down_rgb = CCE(down_out_rgb, sem_batch)
        L_down_hsi = CCE(down_out_hsi, sem_batch)
        L_content = MSE(rgb_content, hsi_content)

        loss_ls = [L_auto_rgb, L_auto_hsi, L_trans_rgb, L_trans_hsi, L_down_rgb, L_down_hsi, L_content]
        temp_loss = 0
        for loss_i, weight_i in zip(loss_ls, log_var_ls):
            temp_loss = temp_loss + torch.exp(-weight_i) * loss_i + weight_i

        total_loss = temp_loss

        total_loss.backward()
        # L_down.backward()
        optimizer.step()
        scheduler.step()

        evaluator.add_batch(sem_batch.cpu().numpy(), np.argmax(down_out_rgb.data.cpu().numpy(), axis=1))

        error_AL.append(total_loss.item())
        error_L_auto_rgb.append(L_auto_rgb.item())
        error_L_trans_rgb.append(L_trans_rgb.item())
        error_L_auto_hsi.append(L_auto_hsi.item())
        error_L_trans_hsi.append(L_trans_hsi.item())
        error_L_down_rgb.append(L_down_rgb.item())
        error_L_down_hsi.append(L_down_hsi.item())
        error_L_content.append(L_content.item())
        ep_lr.append(current_lr)

        # every train_iters//5 save the output for visualization
        if step in list(range(0, 1500, 100)) or step == 1499:
        #     # todo: save result and visualization
        #     epoch_result([row_rgb.cpu().numpy(), row_hsi.cpu().numpy(), rgb_content.detach().cpu().numpy(), hsi_content.detach().cpu().numpy(),
        #                          gen_rgb.detach().cpu().numpy(), gen_hsi.detach().cpu().numpy(), transfer_gen_rgb.detach().cpu().numpy(), transfer_gen_hsi.detach().cpu().numpy(), down_out.detach().cpu().numpy()])

            # down_out = torch.stack([onehot_to_rgb(down_out[i]).unsqueeze(0) for i in range(down_out.shape[0])], dim=0)
            # sem_batch = torch.stack([onehot_to_rgb(sem_batch[i]).unsqueeze(0) for i in range(sem_batch.shape[0])], dim=0)

            down_out_rgb = down_out_rgb.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            down_out_hsi = down_out_hsi.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            sem_batch = sem_batch.detach().cpu().numpy().astype(np.uint8)

            down_out_rgb_temp = []
            down_out_hsi_temp = []
            sem_batch_temp = []

            for img_num in range(batchSize):
                down_out_rgb_temp.append(label2img(down_out_rgb[img_num]))
                down_out_hsi_temp.append(label2img(down_out_hsi[img_num]))
                sem_batch_temp.append(label2img(sem_batch[img_num]))

            down_out_rgb = np.vstack(down_out_rgb_temp) / 255
            down_out_hsi = np.vstack(down_out_hsi_temp) / 255
            sem_batch = np.vstack(sem_batch_temp) / 255

            if batchSize == 1:
                down_out_rgb = down_out_rgb[np.newaxis, :, :, :]
                down_out_hsi = down_out_hsi[np.newaxis, :, :, :]
                sem_batch = sem_batch[np.newaxis, :, :, :]

            down_out_rgb = down_out_rgb.transpose(0, 3, 1, 2)
            down_out_hsi = down_out_hsi.transpose(0, 3, 1, 2)
            sem_batch = sem_batch.transpose(0, 3, 1, 2)

            down_out_rgb = torch.tensor(down_out_rgb, dtype=torch.float32)
            down_out_hsi = torch.tensor(down_out_hsi, dtype=torch.float32)
            sem_batch = torch.tensor(sem_batch, dtype=torch.float32)

            for img, filename in zip([row_rgb, gen_rgb, transfer_gen_rgb, down_out_rgb, down_out_hsi, sem_batch], [f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_row_rgb.jpg',
                                                                                                                   f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_gen_rgb.jpg',
                                                                                                                   f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_transfer_gen_rgb.jpg',
                                                                                                                   f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_down_out_rgb.jpg',
                                                                                                                   f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_down_out_hsi.jpg',
                                                                                                                   f'./results/train_img_new/epoch_{current_epoch}_iters_{step}_sem_batch.jpg']):
                torchvision.utils.save_image(img, filename)

    epoch_loss = np.array([error_AL, error_L_auto_rgb, error_L_trans_rgb, error_L_auto_hsi, error_L_trans_hsi, error_L_down_rgb, error_L_down_hsi, error_L_content, ep_lr]).transpose([1, 0])
    epoch_loss = pd.DataFrame(epoch_loss)
    epoch_loss.columns = ['error_AL', 'error_L_auto_rgb', 'error_L_trans_rgb', 'error_L_auto_hsi', 'error_L_trans_hsi', 'error_L_down_rgb', 'error_L_down_hsi', 'error_L_content', 'ep_lr']
    epoch_loss.to_csv(f'./results/loss_new/train_epoch{current_epoch}.csv', sep=' ')

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    loss_ls = epoch_loss[['error_AL', 'error_L_auto_rgb', 'error_L_trans_rgb', 'error_L_auto_hsi', 'error_L_trans_hsi', 'error_L_down_rgb', 'error_L_down_hsi', 'error_L_content']].mean().to_list()

    print('Train Epoch: {}'.format(current_epoch))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('error_AL: %.3f, error_L_auto_rgb: %.3f, error_L_trans_rgb: %.3f, error_L_auto_hsi: %.3f, error_L_trans_hsi: %.3f, error_L_down_rgb: %.3f, error_L_down_hsi: %.3f, error_L_content: %.3f' % tuple(loss_ls))


    error_AL = []
    error_L_auto_rgb = []
    error_L_trans_rgb = []
    error_L_auto_hsi = []
    error_L_trans_hsi = []
    error_L_down_rgb = []
    error_L_down_hsi = []
    error_L_content = []


    net.eval()
    evaluator.reset()
    for step, (rgb_batch, hsi_batch, sem_batch) in enumerate(valid_dataloader, 0):
        rgb_batch = rgb_batch.type(torch.float32).permute([0, 3, 1, 2]).to(device)
        hsi_batch = hsi_batch.type(torch.float32).permute([0, 3, 1, 2]).to(device)
        sem_batch = sem_batch.type(torch.long).to(device)

        row_rgb, row_hsi, rgb_content, hsi_content, gen_rgb, gen_hsi, transfer_gen_rgb, transfer_gen_hsi, down_out_rgb, down_out_hsi = net(rgb_batch, hsi_batch)

        L_auto_rgb = MSE(row_rgb, gen_rgb)
        L_auto_hsi = MSE(row_hsi, gen_hsi)
        # todo: figure out what trans-loss is
        L_trans_rgb = MSE(row_rgb, transfer_gen_rgb)
        L_trans_hsi = MSE(row_hsi, transfer_gen_hsi)
        L_down_rgb = CCE(down_out_rgb, sem_batch)
        L_down_hsi = CCE(down_out_hsi, sem_batch)
        L_content = MSE(rgb_content, hsi_content)

        loss_ls = [L_auto_rgb, L_auto_hsi, L_trans_rgb, L_trans_hsi, L_down_rgb, L_down_hsi, L_content]
        temp_loss = 0
        for loss_i, weight_i in zip(loss_ls, log_var_ls):
            temp_loss = temp_loss + torch.exp(-weight_i) * loss_i + weight_i

        total_loss = temp_loss

        evaluator.add_batch(sem_batch.cpu().numpy(), np.argmax(down_out_rgb.data.cpu().numpy(), axis=1))

        error_AL.append(total_loss.item())
        error_L_auto_rgb.append(L_auto_rgb.item())
        error_L_trans_rgb.append(L_trans_rgb.item())
        error_L_auto_hsi.append(L_auto_hsi.item())
        error_L_trans_hsi.append(L_trans_hsi.item())
        error_L_down_rgb.append(L_down_rgb.item())
        error_L_down_hsi.append(L_down_hsi.item())
        error_L_content.append(L_content.item())


        # every train_iters//5 save the output for visualization
        if step in list(range(0, 500, 100)) or step == 499:
        #     # todo: save result and visualization
        #     epoch_result([row_rgb.cpu().numpy(), row_hsi.cpu().numpy(), rgb_content.detach().cpu().numpy(), hsi_content.detach().cpu().numpy(),
        #                          gen_rgb.detach().cpu().numpy(), gen_hsi.detach().cpu().numpy(), transfer_gen_rgb.detach().cpu().numpy(), transfer_gen_hsi.detach().cpu().numpy(), down_out.detach().cpu().numpy()])

            # down_out = torch.stack([onehot_to_rgb(down_out[i]).unsqueeze(0) for i in range(down_out.shape[0])], dim=0)
            # sem_batch = torch.stack([onehot_to_rgb(sem_batch[i]).unsqueeze(0) for i in range(sem_batch.shape[0])], dim=0)

            down_out_rgb = down_out_rgb.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            down_out_hsi = down_out_hsi.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            sem_batch = sem_batch.detach().cpu().numpy().astype(np.uint8)

            down_out_rgb_temp = []
            down_out_hsi_temp = []
            sem_batch_temp = []

            for img_num in range(batchSize):
                down_out_rgb_temp.append(label2img(down_out_rgb[img_num]))
                down_out_hsi_temp.append(label2img(down_out_hsi[img_num]))
                sem_batch_temp.append(label2img(sem_batch[img_num]))

            down_out_rgb = np.vstack(down_out_rgb_temp) / 255
            down_out_hsi = np.vstack(down_out_hsi_temp) / 255
            sem_batch = np.vstack(sem_batch_temp) / 255

            if batchSize == 1:
                down_out_rgb = down_out_rgb[np.newaxis, :, :, :]
                down_out_hsi = down_out_hsi[np.newaxis, :, :, :]
                sem_batch = sem_batch[np.newaxis, :, :, :]

            down_out_rgb = down_out_rgb.transpose(0, 3, 1, 2)
            down_out_hsi = down_out_hsi.transpose(0, 3, 1, 2)
            sem_batch = sem_batch.transpose(0, 3, 1, 2)

            down_out_rgb = torch.tensor(down_out_rgb, dtype=torch.float32)
            down_out_hsi = torch.tensor(down_out_hsi, dtype=torch.float32)
            sem_batch = torch.tensor(sem_batch, dtype=torch.float32)

            for img, filename in zip([row_rgb, gen_rgb, transfer_gen_rgb, down_out_rgb, down_out_hsi, sem_batch], [f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_row_rgb.jpg',
                                                                                                                   f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_gen_rgb.jpg',
                                                                                                                   f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_transfer_gen_rgb.jpg',
                                                                                                                   f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_down_out_rgb.jpg',
                                                                                                                   f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_down_out_hsi.jpg',
                                                                                                                   f'./results/val_img_new/epoch_{current_epoch}_iters_{step}_sem_batch.jpg']):
                torchvision.utils.save_image(img, filename)

    epoch_loss = np.array([error_AL, error_L_auto_rgb, error_L_trans_rgb, error_L_auto_hsi, error_L_trans_hsi, error_L_down_rgb, error_L_down_hsi, error_L_content]).transpose([1, 0])
    epoch_loss = pd.DataFrame(epoch_loss)
    epoch_loss.columns = ['error_AL', 'error_L_auto_rgb', 'error_L_trans_rgb', 'error_L_auto_hsi', 'error_L_trans_hsi', 'error_L_down_rgb', 'error_L_down_hsi', 'error_L_content']
    epoch_loss.to_csv(f'./results/loss_new/val_epoch{current_epoch}.csv', sep=' ')

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    loss_ls = epoch_loss[['error_AL', 'error_L_auto_rgb', 'error_L_trans_rgb', 'error_L_auto_hsi', 'error_L_trans_hsi', 'error_L_down_rgb', 'error_L_down_hsi', 'error_L_content']].mean().to_list()

    print('Validation Epoch: {}'.format(current_epoch))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('error_AL: %.3f, error_L_auto_rgb: %.3f, error_L_trans_rgb: %.3f, error_L_auto_hsi: %.3f, error_L_trans_hsi: %.3f, error_L_down_rgb: %.3f, error_L_down_hsi: %.3f, error_L_content: %.3f' % tuple(loss_ls))











