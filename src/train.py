import yaml
import argparse
import torch
import torchvision
import time
from dataset import GameDataset
from model import Model
import numpy as np

start = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/resnet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    with open(opts.config, 'r') as f_in:
        cfg = yaml.load(f_in)
    print(cfg)

    ## Get dataset
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
    data_syn = GameDataset(cfg, normalize)
    dataloader_syn = torch.utils.data.DataLoader(data_syn, num_workers=cfg['NUM_WORKER'],
                                                 batch_size=cfg['BATCH_SIZE'], shuffle=True)
    dataiterator_syn = iter(dataloader_syn)
    print('==> Number of synthetic training images: %d.' % len(data_syn))

    data_test = GameDataset(cfg, normalize, train=False)
    dataloader_test = torch.utils.data.DataLoader(data_test, num_workers=cfg['NUM_WORKER'],
                                                  batch_size=cfg['BATCH_SIZE'], shuffle=False)
    dataiterator_test = iter(dataloader_test)
    print('==> Number of synthetic test images: %d.' % len(data_test))

    if cfg['USE_DA']:
        data_real = torchvision.datasets.ImageFolder(cfg['REAL_DIR'],
                    torchvision.transforms.Compose([
                        torchvision.transforms.Resize((cfg['CROP_SIZE'], cfg['CROP_SIZE'])),
                        torchvision.transforms.ToTensor(),
                        normalize]))
        dataloader_real = torch.utils.data.DataLoader(data_real, num_workers=cfg['NUM_WORKER'],
                                                      batch_size=cfg['BATCH_SIZE'], shuffle=True)
        dataiterator_real = iter(dataloader_real)
        print('==> Number of real training images: %d.' % len(data_real))

    ## Get model
    model = Model()
    model.initialize(cfg)

    ## Training
    for epoch in range(cfg['N_EPOCH']):
        # Get input data, doing this since they of diff. size
        inputs = {}
        while True:
            try:
                inputs['syn'] = next(dataiterator_syn)
            except StopIteration:
                dataiterator_syn = iter(dataloader_syn)
                break
            if cfg['USE_DA']:
                try:
                    inputs['real'] = next(dataiterator_real)
                except StopIteration:
                    dataiterator_real = iter(dataloader_real)
                    inputs['real'] = next(dataiterator_real)

            ## update
            model.set_input(inputs)
            model.optimize()

            ## logging
            model.print_n_log_losses(epoch)
            model.visualize_pred(epoch)

        ## saving
        if (epoch+1) % cfg['SAVE_FREQ'] == 0:
            model.save_networks(epoch)
            # evaluate the model on the test data
            test_inputs = {}
            ratios_11 = []
            ratios_22 = []
            ratios_30 = []
            ratios_45 = []
            batch_sizes = []
            while True:
                try:
                    test_inputs['syn'] = next(dataiterator_test)
                except StopIteration:
                    dataiterator_test = iter(dataloader_test)
                    break

                ## update
                model.set_input(test_inputs)
                test_results = model.test()
                batch_sizes.append(test_results['batch_size'])
                ratios_11.append(test_results['ratio_11'])
                ratios_22.append(test_results['ratio_22'])
                ratios_30.append(test_results['ratio_30'])
                ratios_45.append(test_results['ratio_45'])

            metrics = {}
            metrics['11.25'] = np.average(np.array(ratios_11), weights=np.array(batch_sizes))
            metrics['22.5'] = np.average(np.array(ratios_22), weights=np.array(batch_sizes))
            metrics['30'] = np.average(np.array(ratios_30), weights=np.array(batch_sizes))
            metrics['45'] = np.average(np.array(ratios_45), weights=np.array(batch_sizes))
            print(metrics)

    print('==> Finished Training')
    del dataiterator_syn
    if cfg['USE_DA']:
        del dataiterator_real
end = time.time()
print(end - start)
