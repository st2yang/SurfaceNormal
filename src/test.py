import torch
import torchvision
from dataset import GameDataset
from model import Model
import yaml
import argparse

# 0. Loading configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/resnet_eval.yaml', help='Path to the config file.')
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
model = Model()
model.initialize(cfg)
model.load_networks(cfg['EPOCH_LOAD'])

## Forward model
inputs = {}
count = 0
while True:
    try:
        inputs['syn'] = next(dataiterator_syn)
    except StopIteration:
        dataiterator_syn = iter(dataloader_syn)
        break

    model.set_input(inputs)

    pred = model.forward()

    model.single_test(count, cfg['CKPT_PATH'] + '/ep' + str(cfg['EPOCH_LOAD']))
    count += 1
