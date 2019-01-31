import torch
import torchvision
from dataset import GameDataset
from model import Model
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy

# 0. Loading configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/alexnet.yaml', help='Path to the config file.')
opts = parser.parse_args()
with open(opts.config, 'r') as f_in:
    cfg = yaml.load(f_in)
print(cfg)

# 1. Load testing images from a folder
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
data_syn = GameDataset(cfg, normalize)
dataloader_syn = torch.utils.data.DataLoader(data_syn, num_workers=cfg['NUM_WORKER'],
                                             batch_size=cfg['BATCH_SIZE'], shuffle=False)
dataiterator_syn = iter(dataloader_syn)
print('==> Number of synthetic test images: %d.' % len(data_syn))

# 2. Forward the model
## Get model
epochs = 32
loss_norm_records = []
loss_depth_records = []
loss_records = []
epoch_records = []

## Forward model
inputs = {}

for epoch in range(epochs):
    if epoch % 2 == 1:
        count = 0
        loss_depth = 0
        loss_norm = 0
        total_loss = 0
        model = Model()
        model.initialize(cfg)
        model.load_networks(epoch)
        while True:
            try:
                inputs['syn'] = next(dataiterator_syn)
            except StopIteration:
                dataiterator_syn = iter(dataloader_syn)
                break

            model.set_input(inputs)

            pred = model.forward()

            model.test(count, None)
            count += 1

        loss_depth = model.total_loss_dep / count
        loss_norm = model.total_loss_norm / count
        total_loss = loss_norm + loss_depth
        loss_norm_records.append(loss_norm)
        loss_depth_records.append(loss_depth)
        loss_records.append(total_loss)
        epoch_records.append(epoch)
        print('loss_depth', loss_depth)
        print('loss_norm', loss_norm)
        del model

f, axarr = plt.subplots(2, 2, figsize=(8, 8))
axarr[0, 0].set_title('surface normal loss')
axarr[0, 0].plot(epoch_records, loss_norm_records)
axarr[0, 1].set_title('depth loss')
axarr[0, 1].plot(epoch_records, loss_depth_records)
axarr[1, 0].set_title('total loss')
axarr[1, 0].plot(epoch_records, loss_records)
f.delaxes(axarr[1][1])
plt.show()
