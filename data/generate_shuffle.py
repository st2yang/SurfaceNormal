import numpy as np
import os
import random

#################################
# mixture train
# BATCH_SIZE = 48
#
# # scannet
# data_path = '/home/marsyang/Documents/Dataset/scannet/'
#
# # repeate 3 times
# start_fld = 1
# end_fld = 4
# repetition = 2
# scannet_file_names = []
#
# for o in os.listdir(data_path):
#     sub_dir = os.path.join(data_path, o)
#     if os.path.isdir(sub_dir):
#         for repeat in range(repetition):
#              normal_dir = os.path.join(sub_dir, 'normal')
#              if os.path.isdir(normal_dir):
#                  for filepath in os.listdir(normal_dir):
#                      if filepath.endswith(".png"):
#                          img_names = filepath.split('/')[-1]
#                          img_number = img_names.partition('.')[0].partition('-')[-1]
#                          # leave for testing
#                          if int(img_number) >= 5500:
#                             continue
#                          file_name = 'scannet ' + o + ' ' + str(img_number)
#                          scannet_file_names.append(file_name)
#
# random.shuffle(scannet_file_names)
#
# # nyud
# data_path = '/home/marsyang/Documents/Dataset/nyu_dataset/'
#
# # repeate 5 times
# repetition = 5
# nyud_file_names = []
#
# for _ in range(repetition):
#      ## nyud
#      normal_dir = os.path.join(data_path, 'normal')
#      if os.path.isdir(normal_dir):
#          for filepath in os.listdir(normal_dir):
#              if filepath.endswith(".png"):
#                  img_names = filepath.split('/')[-1]
#                  img_number = img_names.partition('.')[0]
#                  if int(img_number) >= 1300:
#                      continue
#                  file_name = 'nyud ' + str(img_number)
#                  nyud_file_names.append(file_name)
#
# random.shuffle(nyud_file_names)
#
# # scenenet
# data_path = '/home/marsyang/Documents/Dataset/scenenet/train/'
#
# start_fld = 0
# end_fld = 500
# scenenet_file_names = []
#
# for fld in range(start_fld, end_fld):
#      first_fld = int(np.floor(fld / 1000))
#      second_fld = fld
#
#      img_path = data_path + str(first_fld) + '/' + str(second_fld)
#      normal_dir = os.path.join(img_path, 'normal')
#      if os.path.isdir(normal_dir):
#          for filepath in os.listdir(normal_dir):
#              if filepath.endswith(".png"):
#                  img_names = filepath.split('/')[-1]
#                  img_number = img_names.partition('.')[0]
#                  file_name = 'scenenet train/' + str(first_fld) + '/' + str(second_fld) + ' ' + str(img_number)
#                  scenenet_file_names.append(file_name)
#
# random.shuffle(scenenet_file_names)
#
# scanNet_ratio = len(scannet_file_names)
# sceneNet_ratio = len(scenenet_file_names)
# nyud_ratio = len(nyud_file_names)
# total_training_set = scanNet_ratio + sceneNet_ratio + nyud_ratio
# scanNet_ratio = scanNet_ratio / total_training_set * BATCH_SIZE
# sceneNet_ratio = sceneNet_ratio / total_training_set * BATCH_SIZE
# nyud_ratio = nyud_ratio / total_training_set * BATCH_SIZE
# print('scanNet: %d, sceneNet: %d, nyud: %d' % (len(scannet_file_names), len(scenenet_file_names), len(nyud_file_names)))
# print('scanNet: %d, sceneNet: %d, nyud: %d' % (scanNet_ratio, sceneNet_ratio, nyud_ratio))
#
# count_batch = 0
# nyud_idx = 0
# scan_idx = 0
# scene_idx = 0
# with open('mixture_train.txt', 'w') as f:
#   while nyud_idx < len(nyud_file_names) and scan_idx < len(scannet_file_names) and scene_idx < len(scenenet_file_names):
#      if count_batch < nyud_ratio:
#         f.writelines("%s\n" % nyud_file_names[nyud_idx])
#         nyud_idx+=1
#      elif count_batch >= nyud_ratio and count_batch < nyud_ratio + scanNet_ratio:
#         f.writelines("%s\n" % scannet_file_names[scan_idx])
#         scan_idx+=1
#      elif count_batch < BATCH_SIZE:
#         f.writelines("%s\n" % scenenet_file_names[scene_idx])
#         scene_idx+=1
#      else: # count_batch = BATCH_SIZE
#         count_batch = -1
#      count_batch+=1

#################################
# scannet
# data_path = '/home/marsyang/Documents/Dataset/scannet/'
#
# # repeate 3 times
# start_fld = 1
# end_fld = 4
# file_names = []
#
# for o in os.listdir(data_path):
#     sub_dir = os.path.join(data_path, o)
#     if os.path.isdir(os.path.join(data_path, sub_dir)):
#         for fld in range(start_fld, end_fld):
#             first_fld = fld
#             normal_dir = os.path.join(sub_dir, 'normal')
#             if os.path.isdir(normal_dir):
#                 for filepath in os.listdir(normal_dir):
#                     if filepath.endswith(".png"):
#                         img_names = filepath.split('/')[-1]
#                         img_number = img_names.partition('.')[0].partition('-')[-1]
#                         if int(img_number) >= 5500:
#                             continue
#                         file_name = 'scannet ' + o + ' ' + str(img_number)
#                         file_names.append(file_name)
#
# random.shuffle(file_names)
#
# with open('scannet_train.txt', 'w') as f:
#     f.writelines("%s\n" % place for place in file_names)

#################################
# nyud
# data_path = '/home/marsyang/Documents/Dataset/nyu_dataset/'
#
# # repeate 5 times
# start_fld = 1
# end_fld = 6
# file_names = []
#
# for fld in range(start_fld, end_fld):
#     ## nyud
#     first_fld = fld
#     normal_dir = os.path.join(data_path, 'normal')
#     if os.path.isdir(normal_dir):
#         for filepath in os.listdir(normal_dir):
#             if filepath.endswith(".png"):
#                 img_names = filepath.split('/')[-1]
#                 img_number = img_names.partition('.')[0]
#                 if int(img_number) >= 1300:
#                     continue
#                 file_name = 'nyud ' + str(img_number)
#                 file_names.append(file_name)
#
# random.shuffle(file_names)
#
# with open('nyud_train.txt', 'w') as f:
#     f.writelines("%s\n" % place for place in file_names)

#################################
# scenenet
data_path = '/home/marsyang/Documents/Dataset/scenenet/train/'

start_fld = 899
end_fld = 1000
file_names = []

for fld in range(start_fld, end_fld):
    first_fld = int(np.floor(fld / 1000))
    second_fld = fld

    img_path = data_path + str(first_fld) + '/' + str(second_fld)
    normal_dir = os.path.join(img_path, 'normal')
    if os.path.isdir(normal_dir):
        for filepath in os.listdir(normal_dir):
            if filepath.endswith(".png"):
                img_names = filepath.split('/')[-1]
                img_number = img_names.partition('.')[0]
                file_name = 'scenenet train/' + str(first_fld) + '/' + str(second_fld) + ' ' + str(img_number)
                file_names.append(file_name)

# random.shuffle(file_names)

with open('scenenet1000_test.txt', 'w') as f:
    f.writelines("%s\n" % place for place in file_names)


#################################
# real_test_shuffle.txt
# data_path = '/DATA/SyntheticML/nyu_dataset'
#
# file_names = []
# img_path = data_path
# normal_dir = os.path.join(img_path, 'normals')
# if os.path.isdir(normal_dir):
#     for filepath in os.listdir(normal_dir):
#         if filepath.endswith(".png"):
#             img_names = filepath.split('/')[-1]
#             img_number = img_names.partition('.')[0]
#             if int(img_number) < 1150:
#                 continue
#             file_name = 'nyud ' + ' ' + str(img_number)
#             file_names.append(file_name)
#
# with open('real_test_shuffle.txt', 'w') as f:
#     f.writelines("%s\n" % place for place in file_names)

#################################
# mixed_train_shuffle.txt
# rdata_path = '/DATA/SyntheticML/nyu_dataset'
# sdata_path = '/DATA/SyntheticML/scenenet366/train/'
#
# total_fld = 200
# sfile_names = []
# for fld in range(total_fld):
#     first_fld = int(np.floor(fld / 1000))
#     second_fld = fld
#
#     img_path = sdata_path + str(first_fld) + '/' + str(second_fld)
#     normal_dir = os.path.join(img_path, 'normal')
#     if os.path.isdir(normal_dir):
#         for filepath in os.listdir(normal_dir):
#             if filepath.endswith(".png"):
#                 img_names = filepath.split('/')[-1]
#                 img_number = img_names.partition('.')[0]
#                 file_name = 'scenenet train/' + str(first_fld) + '/' + str(second_fld) + ' ' + str(img_number)
#                 sfile_names.append(file_name)
# random.shuffle(sfile_names)
#
# rfile_names = []
# img_path = rdata_path
# normal_dir = os.path.join(img_path, 'normals')
# if os.path.isdir(normal_dir):
#     for filepath in os.listdir(normal_dir):
#         if filepath.endswith(".png"):
#             img_names = filepath.split('/')[-1]
#             img_number = img_names.partition('.')[0]
#             if int(img_number) > 1150:
#                 continue
#             file_name = 'nyud ' + ' ' + str(img_number)
#             rfile_names.append(file_name)
#
# count = 0
# idx = 0
# with open('mixed_train_shuffle.txt', 'w') as f:
#     for place in sfile_names:
#         f.writelines("%s\n" % place)
#         count+=1
#         if(count == 97):
#             count = 0
#             if idx < 1149:
#                 f.writelines("%s\n" % rfile_names[idx])
#                 idx += 1
#                 f.writelines("%s\n" % rfile_names[idx])
#                 idx += 1
#                 f.writelines("%s\n" % rfile_names[idx])
#                 idx += 1
