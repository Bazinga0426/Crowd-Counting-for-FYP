import os
import torch
import numpy as np
import sys

from src.utils import *

# from src import utils


method = 'mcnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

start_step = 0
end_step = 100
lr = 0.001
momentum = 0.9
disp_num = 500
log_interval = 250

# ------------
rand_seed = 99999
best_mae = rand_seed * 100000
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

# load net
net = CrowdCounter()
weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

data_loader = ImgDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImgDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

for epoch in range(start_step, end_step + 1):
    step = -1
    train_loss = 0
    for one_data in data_loader:
        step = step + 1
        im_data = one_data['data']
        gt_data = one_data['gt_density']
        img_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % disp_num == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)
            img_map = img_map.data.cpu().numpy()
            et_count = np.sum(img_map)
            #save_results(im_data, gt_data, img_map, output_dir)
            log_text = 'epoch: %4d, step %4d,  real: %4.1f, pre: %4.1f' % (epoch, step, gt_count, et_count)
            log_print(log_text, color='blue', attrs=['bold'])
            re_cnt = True

        if re_cnt:
            t.tic()
            re_cnt = False

    if (epoch % 20 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
        save_net(save_name, net)
        mae, mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method, dataset_name, epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch, mae, mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
