import torch
from torch.utils import data
import os
import yaml
import argparse
import numpy as np
import matplotlib.pylab as plt
import time


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, results_path):
        self.results_path = results_path
        self.collect_files()

    def collect_files(self):
        self.norm_files = []
        self.depth_files = []
        for filepath in os.listdir(self.results_path):
            if filepath.endswith(".pt"):
                if 'norm' in filepath:
                    self.norm_files.append(filepath)
                elif 'depth' in filepath:
                    self.depth_files.append(filepath)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.norm_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        norm_file = self.norm_files[index]

        # Load data and get label
        cos_val = 1 - torch.load(os.path.join(self.results_path, norm_file)).cpu().detach().numpy()
        # cos_val = torch.min(cos_val, 1 + torch.zeros_like(cos_val))
        # cos_val = torch.max(cos_val, -1.0 + torch.zeros_like(cos_val))

        return cos_val


# Loading configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/resnet_eval.yaml', help='Path to the config file.')
opts = parser.parse_args()
with open(opts.config, 'r') as f_in:
    cfg = yaml.load(f_in)

results_path = cfg['CKPT_PATH']
epoch_num = cfg['EPOCH_LOAD']
device = torch.device('cuda:{}'.format(cfg['GPU_IDS'][0]))

angle_error_tol = 45.0
show_k = 6

save_dir = os.path.join(results_path, 'save')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
results_path = os.path.join(results_path, 'ep' + str(epoch_num))


# Parameters
params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}
training_set = Dataset(results_path)

training_generator = data.DataLoader(training_set, **params)
dataiterator = iter(training_generator)


def angle_error_ratio(angle_degree, base_angle_degree):
    c = torch.gt(base_angle_degree * torch.ones(angle_rad.size(), device=device), angle_degree)
    r = torch.sum(c)
    ratio = torch.div(r, torch.tensor(angle_rad.size()[0] * angle_rad.size()[1], device=device, dtype=torch.float64))
    return ratio

ratios_11 = []
ratios_22 = []
ratios_30 = []
ratios_45 = []
batch_sizes = []
means = []

images_score = np.array([])
images_mean = np.array([])

time_s = time.time()

while True:
    try:
        cos_data = next(dataiterator).to(device)
        angle_rad = torch.acos(cos_data)
        angle_degree = angle_rad / 3.14 * 180
        batch_sizes.append(cos_data.size()[0])
        # ratio metrics
        ratio_11 = angle_error_ratio(angle_degree, 11.25)
        ratio_22 = angle_error_ratio(angle_degree, 22.5)
        ratio_30 = angle_error_ratio(angle_degree, 30)
        ratio_45 = angle_error_ratio(angle_degree, 45)
        ratios_11.append(ratio_11.cpu().detach().numpy())
        ratios_22.append(ratio_22.cpu().detach().numpy())
        ratios_30.append(ratio_30.cpu().detach().numpy())
        ratios_45.append(ratio_45.cpu().detach().numpy())
        # mean metrics
        mean = torch.mean(angle_degree)
        means.append(mean.cpu().detach().numpy())
        # image-wise metrics
        # mean = torch.mean(angle_degree, dim=1)
        # images_mean = np.append(images_mean, mean.cpu().detach().numpy())
        # c = torch.gt(angle_error_tol * torch.ones(angle_rad.size(), device=device), angle_degree)
        # c_img = torch.sum(c, dim=1).double()
        # score = torch.div(c_img, torch.tensor(angle_rad.size()[1], device=device, dtype=torch.float64))
        # images_score = np.append(images_score, score.cpu().detach().numpy())
    except StopIteration:
        break

# statics
# mean
results = {}
results['mean'] = np.average(np.array(means), weights=np.array(batch_sizes))
# results['median'] = np.median(pixels)
results['11.25'] = np.average(np.array(ratios_11), weights=np.array(batch_sizes))
results['22.5'] = np.average(np.array(ratios_22), weights=np.array(batch_sizes))
results['30'] = np.average(np.array(ratios_30), weights=np.array(batch_sizes))
results['45'] = np.average(np.array(ratios_45), weights=np.array(batch_sizes))
print(results)
np.save(os.path.join(save_dir, 'results'), results)

# # plots
# fig = plt.figure()
# plt.hist(pixels, bins=100, density=1, cumulative=True)
# plt.xlim(0, 60)
# plt.ylabel('percentage of pixels')
# plt.xlabel('angle errors')
# plt.show()
# fig.savefig(os.path.join(save_dir, 'pixel.png'))

# fig = plt.figure()
# plt.hist(images_mean, bins=100, density=1, cumulative=True)
# plt.xlim(0, 100)
# plt.ylabel('percentage of images')
# plt.xlabel('mean angle errors of image')
# plt.show()
# fig.savefig(os.path.join(save_dir, 'image.png'))
#
# fig = plt.figure()
# plt.hist(images_score, bins=100, density=1, cumulative=-1)
# plt.xlim(0.0, 1)
# plt.ylabel('percentage of images')
# plt.xlabel('quality of normal prediction (percents of pixels with error less than 45°)')
# plt.show()
# fig.savefig(os.path.join(save_dir, 'quality.png'))
#
# # distribution
# low_k = np.argsort(images_score)[:show_k]
# high_k = np.argsort(-np.array(images_score))[:show_k]
#
# time_f = time.time()
# print('time', time_f-time_s)
