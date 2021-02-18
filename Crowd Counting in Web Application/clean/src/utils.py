import cv2
import os, time
import random
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


class Timer(object):
    def __init__(self):
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.tot_time += self.diff
        self.calls += 1
        self.average_time = self.tot_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ImgDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):

        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2020)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = [i for i in range(0, self.num_samples)]
        if self.pre_load:
            idx = 0
            for filename in self.data_files:

                img = cv2.imread(os.path.join(self.data_path, filename), 0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = int((ht / 4) * 4)
                wd_1 = int((wd / 4) * 4)
                img = cv2.resize(img, (wd_1, ht_1))
                img = img.reshape((1, 1, img.shape[0], img.shape[1]))
                den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(filename)[0] + '.csv'), sep=',', header=None).values#.as_matrix()
                den = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1 // 4
                    ht_1 = ht_1 // 4
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))
                else:
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))

                den = den.reshape((1, 1, den.shape[0], den.shape[1]))
                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['filename'] = filename
                self.blob_list[idx] = blob
                idx = idx + 1
                if idx % 500 == 0:
                    print('Loaded ', idx, '/', self.num_samples, 'files')

            print(' Loading images completed', idx, 'files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                filename = files[idx]
                img = cv2.imread(os.path.join(self.data_path, filename), 0)
                img = img.astype(np.float32, copy=False)
                ht = img.shape[0]
                wd = img.shape[1]
                ht_1 = (ht // 4) * 4
                wd_1 = (wd // 4) * 4
                img = cv2.resize(img, (wd_1, ht_1))
                img = img.reshape((1, 1, img.shape[0], img.shape[1]))
                den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(filename)[0] + '.csv'), sep=',', header=None).as_matrix()
                den = den.astype(np.float32, copy=False)
                if self.gt_downsample:
                    wd_1 = wd_1 // 4
                    ht_1 = ht_1 // 4
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))
                else:
                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))

                den = den.reshape((1, 1, den.shape[0], den.shape[1]))
                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['filename'] = filename

            yield blob

    def get_num_samples(self):
        return self.num_samples


class MCNN(nn.Module):
    '''
    Multi-column CNN
    '''

    def __init__(self, bn=False):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, bn=bn), nn.MaxPool2d(2),Conv2d(16, 32, 7, same_padding=True, bn=bn),nn.MaxPool2d(2),
                                Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, bn=bn), nn.MaxPool2d(2), Conv2d(20, 40, 5, same_padding=True, bn=bn), nn.MaxPool2d(2),
                                Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True, bn=bn), nn.MaxPool2d(2),Conv2d(24, 48, 3, same_padding=True, bn=bn),nn.MaxPool2d(2),
                                Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(filename, net):
    import h5py
    h5f = h5py.File(filename, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(filename, net):
    import h5py
    h5f = h5py.File(filename, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad=False, volatile=True)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        im_data = np_to_variable(im_data, is_cuda=True, is_training=self.training)
        img_map = self.DME(im_data)

        if self.training:
            # print('gt_data')
            gt_data = np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss_mse = self.build_loss(img_map, gt_data)

        return img_map

    def build_loss(self, img_map, gt_data):
        loss = self.loss_fn(img_map, gt_data)
        return loss


def evaluate_model(trained_model, data_loader):
    net = CrowdCounter()
    load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        img_map = net(im_data, gt_data)
        img_map = img_map.data.cpu().numpy()
        real_count = np.sum(gt_data)
        pre_count = np.sum(img_map)
        mae += abs(real_count - pre_count)
        mse += ((real_count - pre_count) * (real_count - pre_count))
    mae = mae / data_loader.get_num_samples()
    mse = np.sqrt(mse / data_loader.get_num_samples())
    return mae, mse









