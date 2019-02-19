import numpy as np
import os
import random

#################################
# nyud
data_path = '/home/marsyang/Documents/Dataset/nyu_dataset/'

# repeate 10 times
start_fld = 1
end_fld = 10
file_names = []

for fld in range(start_fld, end_fld):
    ## nyud
    first_fld = fld
    normal_dir = os.path.join(data_path, 'normal')
    if os.path.isdir(normal_dir):
        for filepath in os.listdir(normal_dir):
            if filepath.endswith(".png"):
                img_names = filepath.split('/')[-1]
                img_number = img_names.partition('.')[0]
                if int(img_number) >= 1300:
                    continue
                file_name = 'nyud ' + str(img_number)
                file_names.append(file_name)

random.shuffle(file_names)

with open('shuffle_debug.txt', 'w') as f:
    f.writelines("%s\n" % place for place in file_names)

#################################
# scenenet
data_path = '/home/marsyang/Documents/Dataset/scenenet/train/'

start_fld = 401
end_fld = 999
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

with open('test_shuffle.txt', 'w') as f:
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
