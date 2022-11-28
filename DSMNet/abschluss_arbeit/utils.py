import numpy as np
import glob
import cv2
import PIL

import torch

from os import path
from PIL import Image
from skimage import io
import os
import torchvision
Image.MAX_IMAGE_PIXELS = 1000000000


def collect_tilenames(mode,dataset):
    all_rgb = []
    all_dsm = []
    all_sem = []
    all_hsi = []

    if dataset == 'Vaihingen':
        trainFrames = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
        valFrames = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
    elif dataset == 'DFC2018':
        trainFrames = ['UH_NAD83_272056_3289689', 'UH_NAD83_272652_3289689', 'UH_NAD83_273248_3289689',
                       'UH_NAD83_273844_3289689']
        valFrames = ['UH_NAD83_271460_3289689', 'UH_NAD83_271460_3290290', 'UH_NAD83_272056_3290290', 'UH_NAD83_272652_3290290', 'UH_NAD83_273248_3290290', 'UH_NAD83_273844_3290290', 'UH_NAD83_274440_3289689', 'UH_NAD83_274440_3290290', 'UH_NAD83_275036_3289689', 'UH_NAD83_275036_3290290']

    if mode == 'train':
        for i in trainFrames:
            if dataset == 'Vaihingen':
                all_rgb.append('../datasets/Vaihingen/RGB/top_mosaic_09cm_area' + str(i) + '.tif')
                all_dsm.append('../datasets/Vaihingen/NDSM/dsm_09cm_matching_area' + str(i) + '.jpg')
                all_sem.append('../datasets/Vaihingen/SEM/top_mosaic_09cm_area' + str(i) + '.tif')
            elif dataset == 'DFC2018':
                all_rgb.append('../datasets/DFC2018/RGB/' + i + '.tif')
                all_dsm.append('../datasets/DFC2018/DSM/' + i + '.tif')
                all_dsm.append('../datasets/DFC2018/DEM/' + i + '.tif')
                all_sem.append('../datasets/DFC2018/SEM/' + i + '.tif')
                all_hsi.append('../datasets/DFC2018/HSI/' + i + '.tif')

    elif mode == 'val':
        for i in valFrames:
            if (dataset == 'Vaihingen'):
                all_rgb.append('../datasets/Vaihingen/RGB/top_mosaic_09cm_area' + str(i) + '.tif')
                all_dsm.append('../datasets/Vaihingen/NDSM/dsm_09cm_matching_area' + str(i) + '.jpg')
                all_sem.append('../datasets/Vaihingen/SEM/top_mosaic_09cm_area' + str(i) + '.tif')
            elif (dataset == 'DFC2018'):
                all_rgb.append('../datasets/DFC2018/RGB/' + i + '.tif')
                all_dsm.append('../datasets/DFC2018/DSM/' + i + '.tif')
                all_dsm.append('../datasets/DFC2018/DEM/' + i + '.tif')
                all_sem.append('../datasets/DFC2018/SEM/' + i + '.tif')
                all_hsi.append('../datasets/DFC2018/HSI/' + i + '.tif')

    return all_rgb, all_dsm, all_sem, all_hsi

def rgb_to_onehot(rgb_image, dataset, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        if (dataset == 'DFC2018'):
            encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 1)) == colormap[i], axis=1).reshape(shape[:2])
        elif (dataset == 'Vaihingen'):
            encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


# todo: make it faster
def onehot_to_rgb(encoded_image):
    ch = encoded_image.shape[0]
    for i in range(ch):
        encoded_image[i, :, :] = encoded_image[i, :, :] * i
    res = encoded_image.sum(dim=0)
    return res


# tensor to PIL Image
def tensor2img(img):
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


# save a set of images
def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))


def genNormals(dsm_tile, mode = 'sobel'):
    if mode == 'gradient':
        zy,zx = np.gradient(dsm_tile)
    elif mode == 'sobel':
        zx = cv2.Sobel(dsm_tile,cv2.CV_64F,1,0,ksize=5)
        zy = cv2.Sobel(dsm_tile,cv2.CV_64F,0,1,ksize=5)

    norm_tile = np.dstack((-zx, -zy, np.ones_like(dsm_tile)))
    n = np.linalg.norm(norm_tile, axis=2)
    norm_tile[:, :, 0] /= n
    norm_tile[:, :, 1] /= n
    norm_tile[:, :, 2] /= n

    norm_tile += 1
    norm_tile /= 2

    return norm_tile

def correctTile(tile):

  tile[tile > 1000] = -123456
  tile[tile == -123456] = np.max(tile)
  tile[tile < -1000] = 123456
  tile[tile == 123456] = np.min(tile)

  return tile


def gaussian_kernel(width, height, sigma=0.2, mu=0.0):
  x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, height))
  d = np.sqrt(x*x+y*y)
  gaussian_k = (np.exp(-((d-mu)**2 / (2.0 * sigma**2)))) / np.sqrt(2 * np.pi * sigma**2)
  return gaussian_k # / gaussian_k.sum()

def sliding_window(image, step, window_size):
  height, width = (image.shape[0], image.shape[1])
  for x in range(0, width, step):
    if x + window_size[0] >= width:
      x = width - window_size[0]
    for y in range(0, height, step):
      if y + window_size[1] >= height:
        y = height - window_size[1]
      yield x, x + window_size[0], y, y + window_size[1]


colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
# 0 – Unclassified
# 1 – Healthy grass
# 2 – Stressed grass
# 3 – Artificial turf
# 4 – Evergreen trees
# 5 – Deciduous trees
# 6 – Bare earth
# 7 – Water
# 8 – Residential buildings
# 9 – Non-residential buildings
# 10 – Roads
# 11 – Sidewalks
# 12 – Crosswalks
# 13 – Major thoroughfares
# 14 – Highways
# 15 – Railways
# 16 – Paved parking lots
# 17 – Unpaved parking lots
# 18 – Cars
# 19 – Trains
# 20 – Stadium seats
def label2img(label):
    h, w = label.shape
    out = (np.ones((h, w, 3)) * 255).astype(np.uint8)
    for class_num, rgb_val in enumerate(colormap):
        out[np.where(label == class_num)] = rgb_val
    return out


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)