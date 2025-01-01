import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from os.path import join
from os import listdir
import time
import zipfile

#file setting
os.makedirs('content/dataset/train/clean',exist_ok=True)
os.makedirs('content/dataset/train/scan',exist_ok=True)
os.makedirs('content/dataset/test/scan',exist_ok=True)
    	
train_clean_zip = zipfile.ZipFile('content/train_clean.zip')
train_clean_zip.extractall('content/dataset/train/clean')
train_clean_zip.close()

train_scan_zip = zipfile.ZipFile('content/train_scan.zip')
train_scan_zip.extractall('content/dataset/train/scan')
train_scan_zip.close()

test_scan_zip = zipfile.ZipFile('content/test_scan.zip')
test_scan_zip.extractall('content/dataset/test/scan')
test_scan_zip.close()