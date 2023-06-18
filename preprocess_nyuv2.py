from scipy.io import loadmat
import numpy as np
import cv2
import os


splits_mat = loadmat('./dataset/nyuv2/raw/splits.mat')
split_train_idx = splits_mat['trainNdxs'].copy()
split_valid_idx = splits_mat['testNdxs'].copy()
# MATLAB index from 1, but numpy index from 0
split_train_idx -= 1
split_valid_idx -= 1

train_idx = -1
valid_idx = -1

# labeled images
if not os.path.exists('./dataset/nyuv2/image'):
    os.makedirs('./dataset/nyuv2/image')
for i in range(1449):
    img_mat = loadmat('./dataset/nyuv2/raw/img_mat/img_{:06d}.mat'.format(i + 1))
    img_tmp = img_mat['img_i']
    img_whole = img_tmp.astype(np.uint8)
    img_save = np.moveaxis(img_whole, -1, 0)  # [C, H, W] for convenience of pytorch data format
    if i in split_train_idx:
        train_idx += 1
        np.save('./dataset/nyuv2/image/train_{}.npy'.format(train_idx), img_save)
    elif i in split_valid_idx:
        valid_idx += 1
        np.save('./dataset/nyuv2/image/valid_{}.npy'.format(valid_idx), img_save)

# labeled semantic segmentation
if not os.path.exists('./dataset/nyuv2/semantic'):
    os.makedirs('./dataset/nyuv2/semantic')
for i in range(795):
    semantic_tmp = cv2.imread('./dataset/nyuv2/raw/nyuv2_train_class13/new_nyu_class13_{:04d}.png'.format(split_train_idx[i][0] + 1), cv2.IMREAD_GRAYSCALE)
    semantic_tmp = semantic_tmp.astype(np.uint8)
    semantic_tmp -= 1
    semantic_save = semantic_tmp
    np.save('./dataset/nyuv2/semantic/train_{}.npy'.format(i), semantic_save)

for i in range(654):
    semantic_tmp = cv2.imread('./dataset/nyuv2/raw/nyuv2_test_class13/new_nyu_class13_{:04d}.png'.format(split_valid_idx[i][0] + 1), cv2.IMREAD_GRAYSCALE)
    semantic_tmp = semantic_tmp.astype(np.uint8)
    semantic_tmp -= 1
    semantic_save = semantic_tmp
    np.save('./dataset/nyuv2/semantic/valid_{}.npy'.format(i), semantic_save)

# labeled depth estimation
train_idx = -1
valid_idx = -1
if not os.path.exists('./dataset/nyuv2/depth'):
    os.makedirs('./dataset/nyuv2/depth')
for i in range(1449):
    depth_mat = loadmat('./dataset/nyuv2/raw/dep_mat/dep_{:06d}.mat'.format(i + 1))
    depth_tmp = depth_mat['dep_i']
    depth_whole = depth_tmp.astype(np.float32)
    depth_save = np.expand_dims(depth_whole, axis=0)  # [C, H, W] for convenience of pytorch data format
    depth_save = np.ascontiguousarray(depth_save)
    if i in split_train_idx:
        train_idx += 1
        np.save('./dataset/nyuv2/depth/train_{}.npy'.format(train_idx), depth_save)
    if i in split_valid_idx:
        valid_idx += 1
        np.save('./dataset/nyuv2/depth/valid_{}.npy'.format(valid_idx), depth_save)
