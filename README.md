# CAENet
Pytorch implementation of "CAENet: Efficient Multi-task Learning for Joint Semantic Segmentation and Depth Estimation", accepted in the Research Track of ECML PKDD 2023.

This repository contains codes for dataset preprocessing and model training. For reimplementation, you can follow this instruction :)

## 1. Preprocess Dataset
### 1.1 NYUv2
#### 1.1.1 Download and Extract
All the raw data needs to be downloaded in ./dataset/nyuv2/raw/.  
Please download [nyu_depth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat).  
Please download [nyuv2_train_class13.tgz](https://github.com/ankurhanda/nyuv2-meta-data/blob/master/train_labels_13/nyuv2_train_class13.tgz) and extract it as nyuv2_train_class13.  
Please download [nyuv2_test_class13.tgz](https://github.com/ankurhanda/nyuv2-meta-data/blob/master/test_labels_13/nyuv2_test_class13.tgz) and extract it as nyuv2_test_class13.
#### 1.1.2 Preprocess
Enter the directory ./dataset/nyuv2/raw/, and run `matlab -nodisplay -r extract_single_mat` to extract single files for images and depth labels.  
Go back to ./, and run `python preprocess_nyuv2.py` to prepare the dataset.
### 1.2 CityScapes
#### 1.2.1 Download and Extract
All the raw data needs to be downloaded in ./dataset/city/raw/.  
Please download [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) and extract it just in the current folder.  
Please download [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and extract it just in the current folder.  
Please download [disparity_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=7) and extract it just in the current folder.  
Please download [camera_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=8) and extract it just in the current folder.
#### 1.2.2 Preprocess
Go back to ./, and run `python preprocess_city.py` to prepare the dataset.  
## 2. Pretrained Weights on ImageNet
All the pretrained weights on ImageNet for the backbone need to be downloaded in ./pretrained_weights/.
### 2.1 MobileNetV2
Please download [mobilenetv2-e6e8dd43.pth](https://cloudstor.aarnet.edu.au/plus/s/uRgFbkaRjD3qOg5/download).
### 2.2 ResNet18
Please download [resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth).
### 2.3 STDCNet
Please download [STDCNet1446_76.47.tar](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1).  
## 3. Train and Evaluate
Run `sh run.sh` can get the model trained and evaluated.  
The input args in run.sh can be changed to test for different backbones and datasets.  
Several training log examples are provided in ./logs/.  
## Acknowledgement
We truly appreciate the following researchers for their excellent work and code.
+ Nekrasov et al. [multi-task-refinenet](https://github.com/drsleep/multi-task-refinenet)
+ Fan et al. [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg)
+ Liu et al. [SAMNet](https://github.com/yun-liu/FastSaliency)
+ Liu et al. [MTAN](https://github.com/lorenmt/mtan)
