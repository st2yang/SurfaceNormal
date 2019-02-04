# generate statistics and plots from the result files

import torch
import time
import numpy as np
import matplotlib.pylab as plt
import os
import yaml
import argparse

# Loading configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/resnet_eval.yaml', help='Path to the config file.')
opts = parser.parse_args()
with open(opts.config, 'r') as f_in:
    cfg = yaml.load(f_in)

results_path = cfg['CKPT_PATH']
epoch_num = cfg['EPOCH_LOAD']

angle_error_tol = 45
show_k = 6

save_dir = os.path.join(results_path, 'save')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
results_path = os.path.join(results_path, 'ep' + str(epoch_num))

pixels = np.array([])
images_mean = []
images_score = []
images_path = []


time_s = time.time()
if os.path.isdir(results_path):
    for filepath in os.listdir(results_path):
        if filepath.endswith(".pt") and 'norm' in filepath:
            cos_val = 1 - torch.load(os.path.join(results_path, filepath)).cpu().detach().numpy()
            cos_val = np.minimum(cos_val, 1.0)
            cos_val = np.maximum(cos_val, -1.0)
            a = np.degrees(np.arccos(cos_val))
            # process on an image
            score = np.sum(np.less(a, angle_error_tol)) / len(a)
            images_score.append(score)
            images_path.append(os.path.join(results_path, filepath))
            images_mean.append(np.mean(a))
            # process all pixels
            pixels = np.append(pixels, a)

# statics
# mean
results = {}
results['mean'] = np.mean(pixels)
results['median'] = np.median(pixels)
results['11.25'] = np.sum(np.less(pixels, 11.25)) / len(pixels)
results['22.5'] = np.sum(np.less(pixels, 22.5)) / len(pixels)
results['30'] = np.sum(np.less(pixels, 30)) / len(pixels)
results['45'] = np.sum(np.less(pixels, 45)) / len(pixels)
np.save(os.path.join(save_dir, 'results'), results)

# plots
fig = plt.figure()
plt.hist(pixels, bins=100, density=1, cumulative=True)
plt.xlim(0, 60)
plt.ylabel('percentage of pixels')
plt.xlabel('angle errors')
plt.show()
fig.savefig(os.path.join(save_dir, 'pixel.png'))

fig = plt.figure()
plt.hist(images_mean, bins=100, density=1, cumulative=True)
plt.xlim(0, 100)
plt.ylabel('percentage of images')
plt.xlabel('mean angle errors of image')
plt.show()
fig.savefig(os.path.join(save_dir, 'image.png'))

fig = plt.figure()
plt.hist(images_score, bins=100, density=1, cumulative=-1)
plt.xlim(0.0, 1)
plt.ylabel('percentage of images')
plt.xlabel('quality of normal prediction (percents of pixels with error less than 45°)')
plt.show()
fig.savefig(os.path.join(save_dir, 'quality.png'))

# distribution
low_k = np.argsort(images_score)[:show_k]
high_k = np.argsort(-np.array(images_score))[:show_k]

time_f = time.time()
print('time', time_f-time_s)
