import yaml
import argparse
import torch
import torchvision
import time
from dataset import GameDataset
from model import Model
import numpy as np
import os
import matplotlib.pylab as plt

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

    data_test = GameDataset(cfg, normalize, train=False, real_data=False)
    dataloader_test = torch.utils.data.DataLoader(data_test, num_workers=cfg['NUM_WORKER'],
                                                  batch_size=cfg['BATCH_SIZE'], shuffle=False)
    dataiterator_test = iter(dataloader_test)
    print('==> Number of synthetic test images: %d.' % len(data_test))

    real_data_test = GameDataset(cfg, normalize, train=False, real_data=True)
    realdataloader_test = torch.utils.data.DataLoader(real_data_test, num_workers=cfg['NUM_WORKER'],
                                                  batch_size=1, shuffle=False)
    realdataiterator_test = iter(realdataloader_test)
    print('==> Number of nyud test images: %d.' % len(real_data_test))

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
    model.load_networks(which_epoch=cfg['EPOCH_LOAD'])
    metrics_dir = os.path.join(model.save_dir, 'metrics')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)

    ## Training
    for epoch in range(cfg['EPOCH_LOAD'] + 1, cfg['N_EPOCH']):
        print('--------------')
        print('training epoch', epoch)
        # Get input data, doing this since they of diff. size
        inputs = {}
        while True:
            try:
                inputs['syn'] = next(dataiterator_syn)
            except StopIteration:
                dataiterator_syn = iter(dataloader_syn)
                break

            ## update
            model.set_input(inputs)
            model.optimize()

            ## logging
            # model.print_n_log_losses(epoch)
            # model.visualize_pred(epoch)

        ## saving
        if (epoch+1) % cfg['SAVE_FREQ'] == 0:
            model.save_networks(epoch)
            # evaluate the model on the test data
            count = 0
            realdata_inputs = {}
            ratios_11 = []
            ratios_22 = []
            ratios_30 = []
            ratios_45 = []
            batch_sizes = []
            pixel_errors = np.array([])

            while True:
                try:
                    realdata_inputs['syn'] = next(realdataiterator_test)
                except StopIteration:
                    realdataiterator_test = iter(realdataloader_test)
                    break

                ## save to file
                model.set_input(realdata_inputs)
                real_test_results = model.test()
                ratios_11.append(real_test_results['ratio_11'])
                ratios_22.append(real_test_results['ratio_22'])
                ratios_30.append(real_test_results['ratio_30'])
                ratios_45.append(real_test_results['ratio_45'])
                pixel_errors = np.append(pixel_errors, real_test_results['pixel_error'])
                model.out_logic_map(epoch_num=epoch, img_num=count)
                count += 1
            metrics = {}
            metrics['mean'] = np.mean(pixel_errors)
            metrics['median'] = np.median(pixel_errors)
            metrics['11.25'] = np.average(np.array(ratios_11))
            metrics['22.5'] = np.average(np.array(ratios_22))
            metrics['30'] = np.average(np.array(ratios_30))
            metrics['45'] = np.average(np.array(ratios_45))
            np.save(os.path.join(metrics_dir, 'results_real_ep{}'.format(epoch)), metrics)
            print('real data nyud metrics: ', metrics)

            test_inputs = {}
            ratios_11 = []
            ratios_22 = []
            ratios_30 = []
            ratios_45 = []
            batch_sizes = []
            image_scores = np.array([])
            image_means = np.array([])
            pixel_errors = np.array([])

            while True:
                try:
                    test_inputs['syn'] = next(dataiterator_test)
                except StopIteration:
                    dataiterator_test = iter(dataloader_test)
                    break
                model.set_input(test_inputs)
                test_results = model.test()
                batch_sizes.append(test_results['batch_size'])
                ratios_11.append(test_results['ratio_11'])
                ratios_22.append(test_results['ratio_22'])
                ratios_30.append(test_results['ratio_30'])
                ratios_45.append(test_results['ratio_45'])
                pixel_errors = np.append(pixel_errors, test_results['pixel_error'])
                image_means = np.append(image_means, test_results['image_mean'])
                image_scores = np.append(image_scores, test_results['image_score'])

            metrics = {}
            metrics['mean'] = np.mean(pixel_errors)
            metrics['median'] = np.median(pixel_errors)
            metrics['11.25'] = np.average(np.array(ratios_11), weights=np.array(batch_sizes))
            metrics['22.5'] = np.average(np.array(ratios_22), weights=np.array(batch_sizes))
            metrics['30'] = np.average(np.array(ratios_30), weights=np.array(batch_sizes))
            metrics['45'] = np.average(np.array(ratios_45), weights=np.array(batch_sizes))
            np.save(os.path.join(metrics_dir, 'results_synthetic_ep{}'.format(epoch)), metrics)
            print('synthetic data metrics: ', metrics)

            # plots
            fig = plt.figure()
            plt.hist(pixel_errors, bins=1000, density=1)
            plt.xlim(0, 60)
            plt.ylabel('percentage of pixels')
            plt.xlabel('angle errors')
            fig.savefig(os.path.join(metrics_dir, 'histogram_ep{}.png'.format(epoch)))
            fig.clf()

            fig = plt.figure()
            plt.hist(image_means, bins=100, density=1, cumulative=True)
            plt.xlim(0, 100)
            plt.ylabel('percentage of images')
            plt.xlabel('mean angle errors of image')
            fig.savefig(os.path.join(metrics_dir, 'image_ep{}.png'.format(epoch)))
            fig.clf()

            fig = plt.figure()
            plt.hist(image_scores, bins=100, density=1, cumulative=-1)
            plt.xlim(0.0, 1)
            plt.ylabel('percentage of images')
            plt.xlabel('quality of normal prediction (percents of pixels with error less than 45°)')
            fig.savefig(os.path.join(metrics_dir, 'quality_ep{}.png'.format(epoch)))
            fig.clf()

            # TODO: close all saved figures for memory efficiency

    print('==> Finished Training')
    del dataiterator_syn
    if cfg['USE_DA']:
        del dataiterator_real
end = time.time()
print(end - start)
