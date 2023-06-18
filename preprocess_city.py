import glob
import os
import numpy as np
from PIL import Image
import json
import torch
import torch.nn.functional as F
import cv2

# labeled images
if not os.path.exists('./dataset/city/image'):
    os.makedirs('./dataset/city/image')

directory_list = glob.glob('./dataset/city/raw/leftImg8bit/train/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*.png'))
    file_list.sort()
    for f in file_list:
        idx += 1
        img = Image.open(f)
        img = img.resize((1024, 512), Image.BILINEAR)
        img_np = np.array(img)
        img_np = img_np.astype(np.uint8)
        img_np = np.moveaxis(img_np, -1, 0)  # (3, 512, 1024)
        np.save('./dataset/city/image/train_{}.npy'.format(idx), img_np)

directory_list = glob.glob('./dataset/city/raw/leftImg8bit/val/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*.png'))
    file_list.sort()
    for f in file_list:
        idx += 1
        img = Image.open(f)
        img = img.resize((1024, 512), Image.BILINEAR)
        img_np = np.array(img)
        img_np = img_np.astype(np.uint8)
        img_np = np.moveaxis(img_np, -1, 0)  # (3, 512, 1024)
        np.save('./dataset/city/image/valid_{}.npy'.format(idx), img_np)

# labeled semantic segmentation
if not os.path.exists('./dataset/city/semantic'):
    os.makedirs('./dataset/city/semantic')

label_mapping = [
    255, 255, 255, 255, 255,
    255, 255, 0, 1, 255,
    255, 2, 3, 4, 255,
    255, 255, 5, 255, 6,
    7, 8, 9, 10, 11,
    12, 13, 14, 15, 255,
    255, 16, 17, 18
]
label_mapping = np.array(label_mapping)

directory_list = glob.glob('./dataset/city/raw/gtFine/train/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*_labelIds.png'))
    file_list.sort()
    for f in file_list:
        idx += 1
        img = Image.open(f)
        img = img.resize((1024, 512), Image.NEAREST)  # must use nearest for semantic segmentation
        img_np = np.array(img)
        img_np = label_mapping[img_np]
        img_np = img_np.astype(np.uint8)
        np.save('./dataset/city/semantic/train_{}.npy'.format(idx), img_np)

directory_list = glob.glob('./dataset/city/raw/gtFine/val/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*_labelIds.png'))
    file_list.sort()
    for f in file_list:
        idx += 1
        img = Image.open(f)
        img = img.resize((1024, 512), Image.NEAREST)  # must use nearest for semantic segmentation
        img_np = np.array(img)
        img_np = label_mapping[img_np]
        img_np = img_np.astype(np.uint8)
        np.save('./dataset/city/semantic/valid_{}.npy'.format(idx), img_np)

# labeled depth estimation
if not os.path.exists('./dataset/city/depth'):
    os.makedirs('./dataset/city/depth')

directory_list = glob.glob('./dataset/city/raw/camera/train/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*.json'))
    file_list.sort()
    for f in file_list:
        idx += 1
        with open(f, 'r') as file:
            load_dict = json.load(file)
            baseline = load_dict['extrinsic']['baseline']
            fx = load_dict['intrinsic']['fx']

        city_name = d.split('/')[-1]
        img_name = f.split('/')[-1]
        img_name = img_name[:-11]
        disp_file = './dataset/city/raw/disparity/train/{}/{}disparity.png'.format(city_name, img_name)
        disparity = cv2.imread(disp_file, cv2.IMREAD_UNCHANGED)
        disparity = disparity.astype(np.float32)
        mask = disparity > 0
        disparity[mask] = (disparity[mask]-1)/256
        depth = np.zeros_like(disparity)
        mask_new = disparity > 0
        depth[mask_new] = baseline*fx/disparity[mask_new]
        depth[depth > 100] = 0  # =0 for ground truth label large value cutout
        depth = np.expand_dims(depth, axis=0)
        depth_torch = torch.from_numpy(depth).cuda()
        depth_torch = F.interpolate(depth_torch[None, :, :, :], size=(512, 1024), mode='bilinear', align_corners=True).squeeze(0)
        depth = depth_torch.cpu().numpy()
        depth = depth.astype(np.float32)
        np.save('./dataset/city/depth/train_{}.npy'.format(idx), depth)

directory_list = glob.glob('./dataset/city/raw/camera/val/*')
directory_list.sort()
idx = -1
for d in directory_list:
    file_list = glob.glob(os.path.join(d, '*.json'))
    file_list.sort()
    for f in file_list:
        idx += 1
        with open(f, 'r') as file:
            load_dict = json.load(file)
            baseline = load_dict['extrinsic']['baseline']
            fx = load_dict['intrinsic']['fx']

        city_name = d.split('/')[-1]
        img_name = f.split('/')[-1]
        img_name = img_name[:-11]
        disp_file = './dataset/city/raw/disparity/val/{}/{}disparity.png'.format(city_name, img_name)
        disparity = cv2.imread(disp_file, cv2.IMREAD_UNCHANGED)
        disparity = disparity.astype(np.float32)
        mask = disparity > 0
        disparity[mask] = (disparity[mask]-1)/256
        depth = np.zeros_like(disparity)
        mask_new = disparity > 0
        depth[mask_new] = baseline*fx/disparity[mask_new]
        depth[depth > 100] = 0  # =0 for ground truth label large value cutout
        depth = np.expand_dims(depth, axis=0)
        depth_torch = torch.from_numpy(depth).cuda()
        depth_torch = F.interpolate(depth_torch[None, :, :, :], size=(512, 1024), mode='bilinear', align_corners=True).squeeze(0)
        depth = depth_torch.cpu().numpy()
        depth = depth.astype(np.float32)
        np.save('./dataset/city/depth/valid_{}.npy'.format(idx), depth)
