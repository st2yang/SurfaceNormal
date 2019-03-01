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
fig = plt.figure()


def evaluation_metrics(model, data_name, dataloader_test, metrics_dir, epoch):
    inputs = {}
    ratios_11 = []
    ratios_22 = []
    ratios_30 = []
    ratios_45 = []
    batch_sizes = []
    image_scores = np.array([])
    image_means = np.array([])
    pixel_errors = np.array([])

    dataiterator_test = iter(dataloader_test)

    while True:
        try:
            inputs['syn'] = next(dataiterator_test)
        except StopIteration:
            break
        model.set_input(inputs)
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
    np.save(os.path.join(metrics_dir, 'results_{}_ep{}'.format(data_name, epoch)), metrics)
    print(data_name, ' test metrics: ', metrics)

    # plots
    plt.hist(pixel_errors, bins=1000, density=1)
    plt.xlim(0, 60)
    plt.ylabel('percentage of pixels')
    plt.xlabel('angle errors')
    fig.savefig(os.path.join(metrics_dir, 'histogram_{}_ep{}.png'.format(data_name, epoch)))
    fig.clf()

    plt.hist(image_means, bins=100, density=1, cumulative=True)
    plt.xlim(0, 100)
    plt.ylabel('percentage of images')
    plt.xlabel('mean angle errors of image')
    fig.savefig(os.path.join(metrics_dir, 'image_{}_ep{}.png'.format(data_name, epoch)))
    fig.clf()

    # TODO: to check this metric
    plt.hist(image_scores, bins=100, density=1, cumulative=-1)
    plt.xlim(0.0, 1)
    plt.ylabel('percentage of images')
    plt.xlabel('quality of normal prediction (percents of pixels with error less than 45°)')
    fig.savefig(os.path.join(metrics_dir, 'quality_{}_ep{}.png'.format(data_name, epoch)))
    fig.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/resnet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    with open(opts.config, 'r') as f_in:
        cfg = yaml.load(f_in)
    print(cfg)

    ## Get dataset
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
    data_train = GameDataset(cfg, normalize)
    dataloader_train = torch.utils.data.DataLoader(data_train, num_workers=cfg['NUM_WORKER'],
                                                   batch_size=cfg['BATCH_SIZE'], shuffle=True)
    dataiterator_train = iter(dataloader_train)
    print('==> Number of training images: %d.' % len(data_train))

    # nyud_test = GameDataset(cfg, normalize, train=False, test_data='nyud')
    # dataloader_nyud_test = torch.utils.data.DataLoader(nyud_test, num_workers=cfg['NUM_WORKER'],
    #                                                    batch_size=1, shuffle=False)
    # dataiterator_nyud_test = iter(dataloader_nyud_test)
    # print('==> Number of nyud test images: %d.' % len(nyud_test))

    scenenet_test = GameDataset(cfg, normalize, train=False, test_data='scenenet')
    dataloader_scenenet_test = torch.utils.data.DataLoader(scenenet_test, num_workers=cfg['NUM_WORKER'],
                                                           batch_size=cfg['BATCH_SIZE'], shuffle=False)
    print('==> Number of scenenet test images: %d.' % len(scenenet_test))

    # scannet_test = GameDataset(cfg, normalize, train=False, test_data='scannet')
    # dataloader_scannet_test = torch.utils.data.DataLoader(scannet_test, num_workers=cfg['NUM_WORKER'],
    #                                                       batch_size=cfg['BATCH_SIZE'], shuffle=False)
    # print('==> Number of scannet test images: %d.' % len(scannet_test))

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
    # model.load_networks(which_epoch=cfg['EPOCH_LOAD'])
    metrics_dir = os.path.join(model.save_dir, 'metrics')
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)

    ## Training
    for epoch in range(cfg['EPOCH_LOAD'] + 1, cfg['N_EPOCH']):
        print('--------------')
        print('training epoch', epoch)
        # Get input data, doing this since they of diff. size
        train_inputs = {}
        while True:
            try:
                train_inputs['syn'] = next(dataiterator_train)
            except StopIteration:
                dataiterator_train = iter(dataloader_train)
                break

            ## update
            model.set_input(train_inputs)
            model.optimize()

            ## logging
            # model.print_n_log_losses(epoch)
            # model.visualize_pred(epoch)

        ## saving
        if (epoch+1) % cfg['SAVE_FREQ'] == 0:
            model.save_networks(epoch)
            # evaluate the model on the nyud test data
            # nyud_count = 0
            # nyud_inputs = {}
            # nyud_ratios_11 = []
            # nyud_ratios_22 = []
            # nyud_ratios_30 = []
            # nyud_ratios_45 = []
            # nyud_batch_sizes = []
            # nyud_pixel_errors = np.array([])
            #
            # while True:
            #     try:
            #         nyud_inputs['syn'] = next(dataiterator_nyud_test)
            #     except StopIteration:
            #         dataiterator_nyud_test = iter(dataloader_nyud_test)
            #         break
            #
            #     ## save to file
            #     model.set_input(nyud_inputs)
            #     nyud_test_results = model.test()
            #     nyud_ratios_11.append(nyud_test_results['ratio_11'])
            #     nyud_ratios_22.append(nyud_test_results['ratio_22'])
            #     nyud_ratios_30.append(nyud_test_results['ratio_30'])
            #     nyud_ratios_45.append(nyud_test_results['ratio_45'])
            #     nyud_pixel_errors = np.append(nyud_pixel_errors, nyud_test_results['pixel_error'])
            #     model.out_logic_map(epoch_num=epoch, img_num=nyud_count)
            #     nyud_count += 1
            # nyud_metrics = {}
            # nyud_metrics['mean'] = np.mean(nyud_pixel_errors)
            # nyud_metrics['median'] = np.median(nyud_pixel_errors)
            # nyud_metrics['11.25'] = np.average(np.array(nyud_ratios_11))
            # nyud_metrics['22.5'] = np.average(np.array(nyud_ratios_22))
            # nyud_metrics['30'] = np.average(np.array(nyud_ratios_30))
            # nyud_metrics['45'] = np.average(np.array(nyud_ratios_45))
            # np.save(os.path.join(metrics_dir, 'results_nyud_ep{}'.format(epoch)), nyud_metrics)
            # print('nyud test metrics: ', nyud_metrics)

            evaluation_metrics(model, 'scenenet', dataloader_scenenet_test, metrics_dir, epoch)
            # evaluation_metrics(model, 'scannet', dataloader_scannet_test, metrics_dir, epoch)

    print('==> Finished Training')
    del dataiterator_train
    if cfg['USE_DA']:
        del dataiterator_real
end = time.time()
print(end - start)
