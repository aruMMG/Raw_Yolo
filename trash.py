import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
# import matplotlib
# print(matplotlib.get_backend())
# matplotlib.use("TkAgg")
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
import cv2


read_add = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/val/'
BIT8  = 2 ** 8
BIT16 = 2 ** 16
BIT24 = 2 ** 24
height = 1856
width = 2880

#
# def imshow_sep(image, title="image"):
#     plt.figure()
#     plt.title(title)
#     plt.imshow(image)

def imshow(image, title="image"):
    print('showing!')
    plt.title(title)
    plt.imshow(image)


def load_image_raw(self, i):
    # Mirror of load_image() but for RAW; uses same target img_size logic
    f_raw = self.raw_files[i]
    im = cv2.imread(f_raw, cv2.IMREAD_UNCHANGED)  # RAW could be 1ch or 16-bit
    assert im is not None, f'RAW image Not Found {f_raw}'
    # If single channel, expand to HxWx1 to keep transforms generic
    if im.ndim == 2:
        im = im[..., None]
    h0, w0 = im.shape[:2]
    r = self.img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return im, (h0, w0), im.shape[:2]

def read_raw_24b(file_path, img_shape=(1, 1, height, width), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    print(raw_data.shape)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    print(raw_data.shape)
    # raw_data = raw_data.reshape(img_shape).astype(np.uint8)
    raw_data = raw_data.reshape(height, width)/2**16

    return raw_data

def read_raw(file_path, img_shape=(1, 1, height, width), read_type=np.uint8):
    raw = np.fromfile(file_path, dtype=np.uint8)
    raw = raw.reshape(1856, 2880, 3).astype(np.float32)
    raw = np.split(raw, 3, axis=2)
    raw = (raw[0] + raw[1] * BIT8 + raw[2] * BIT16)
    raw = raw / (BIT24 - 1) * BIT16  # norm to range [0, 1]
    raw = np.clip(raw, 0, 65535).astype(np.uint16)
    return raw



i = 1
print(f'{read_add}*.raw')
for name in glob.glob(f'{read_add}day*.raw'):
    print(f'{i} of 16000 ---- {name}')
    i+=1
    raw_image = read_raw_24b(name)
    # print(raw_image)
    print(f'shape {raw_image.shape}')
    imshow(raw_image)
    # imageio.imwrite('test.tiff', raw_image, format='TIFF')
    # raw = rawpy.imread('test.tiff')
    # # im = raw.postprocess(half_size=True, output_bps=16).astype(np.uint16)/2**16
    # im = raw.postprocess(half_size=True, output_bps=16, bright=3, gamma=(2.222, 4.5), dcb_enhance=True, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(2), use_auto_wb=True).astype(np.uint16) / 2 ** 16
    # # tonemap_mantiuk = cv2.createTonemapMantiuk()
    # # # Apply the tonemap operator to the HDR image
    # # ldr_image = tonemap_mantiuk.process(im.astype(np.float32))
    # # imshow_sep(ldr_image)
    # # imshow_sep(im)
    # # print(f'after write {im.shape}')
    # # cv2.imwrite(f'{save_add}{name.split("/")[-1].split(".")[0]}.jpg', cv2.cvtColor(im.astype('float32')**(1/2.2)*255.0, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(f'{save_add}{name.split("/")[-1].split(".")[0]}.jpg', cv2.cvtColor(im.astype('float32')*255.0, cv2.COLOR_BGR2RGB))
    plt.show()



