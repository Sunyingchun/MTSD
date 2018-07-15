import os
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import random
import numpy as np
from PIL import Image

from loader.augment import augmentation
import config.config as config

class cityscapesLoader(Dataset):

    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, listPath, split = 'train', imgSize = 416, batchSize = 8, is_transform = True, is_augmentation = True, shuffle = True, randomResize = True):
    
        self.imgSize = imgSize
        self.oriImgSize = imgSize
        self.batchSize = batchSize
        self.split = split
        self.is_transform = is_transform
        self.is_augmentation = is_augmentation
        self.shuffle = shuffle
        self.randomResize = randomResize

        # read files
        with open(listPath, 'r') as file:
            self.imgFiles = file.readlines()

        # detection
        self.maxObjects = 50
        self.jitter = config.jitter
        self.hue =  config.hue
        self.saturation =  config.saturation
        self.exposure =  config.exposure

        # segmentation
        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19))) 

    def __len__(self):
        """__len__"""
        return len(self.imgFiles)

    def __getitem__(self, index):
        if self.split == "train" and self.randomResize == True and index % self.batchSize == 0:
            width = (random.randint(5, 9)+10) * 32
            self.imgSize = width

        if self.split == "train" and self.shuffle:
            random.shuffle(self.imgFiles)

        imgPath = self.imgFiles[index % len(self.imgFiles)].rstrip('\n')
        img = Image.open(imgPath)

        labelPath = imgPath.replace('JPEGImages', 'labels').replace('png', 'txt')
        labels = np.loadtxt(labelPath).reshape(-1, 5)

        segPath = imgPath.replace('JPEGImages', 'SegmentLabel').replace('leftImg8bit', 'gtFine_labelIds')
        segMap = Image.open(segPath)

        if self.split == "train" and self.is_augmentation:
            img, labels, segMap = augmentation(img, labels, segMap, (self.imgSize, self.imgSize), self.jitter, self.hue, self.saturation, self.exposure)
 
        #Image to numpy
        img = np.array(img)
        segMap = np.array(segMap)

        if self.is_transform:
            in_, labels, seg_ = self.transform(img, labels, segMap)

        # Fill matrix
        filled_labels = np.zeros((self.maxObjects, 5))
        filled_labels[range(len(labels))[:self.maxObjects]] = labels[:self.maxObjects]
        filled_labels = torch.from_numpy(filled_labels)

        # encode segMap
        seg_ = self.encode_segmap(np.array(seg_, dtype=np.uint8))
        seg_ = torch.from_numpy(seg_).long()

        return in_, filled_labels, seg_

    def transform(self, img, labels, seg):
        #################
        #image
        #################
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        in_ = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = in_.shape
        # Resize and normalize
        in_ = cv2.resize(in_, (self.imgSize, self.imgSize))
        # Channels-first
        #in_ = in_[:,:,::-1]
        in_ = np.transpose(in_, (2, 0, 1))
        # As pytorch tensor
        in_ = torch.from_numpy(in_).float()

        #################
        #label
        #################
        x1 = w * (labels[:, 1] - labels[:, 3]/2)
        y1 = h * (labels[:, 2] - labels[:, 4]/2)
        x2 = w * (labels[:, 1] + labels[:, 3]/2)
        y2 = h * (labels[:, 2] + labels[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= float(w) / float(padded_w)
        labels[:, 4] *= float(h) / float(padded_h)

        #################
        #segMap
        #################        
        padSeg = ((pad1, pad2), (0, 0)) if h <= w else ((0, 0), (pad1, pad2))
        seg_ = np.pad(seg, padSeg, 'constant', constant_values=0)
        # Resize and normalize
        seg_ = cv2.resize(seg_, (self.imgSize, self.imgSize))

        return in_, labels, seg_

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb

    def decode_labels(self, img, labels):

        h, w, _ = img.shape

        x1 = w * (labels[:, 1] - labels[:, 3]/2)
        y1 = h * (labels[:, 2] - labels[:, 4]/2)
        x2 = w * (labels[:, 1] + labels[:, 3]/2)
        y2 = h * (labels[:, 2] + labels[:, 4]/2)

        return x1, y1, x2, y2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    dst = cityscapesLoader("/home/wfw/data/VOCdevkit/cityscapesMTSD/val.txt", split = 'train', batchSize = 8, imgSize = 608)

    bs = 8
    trainloader = DataLoader(dst, batch_size=bs, num_workers=8)
    for i, data in enumerate(trainloader):
        imgs, labels, segMaps = data
        for img, label, segMap in zip(imgs, labels, segMaps):
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0)).copy()
            seg = dst.decode_segmap(segMap.numpy())
            xmins, ymins, xmaxs, ymaxs = dst.decode_labels(seg, label.numpy())

            for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.rectangle(seg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            cv2.imshow('seg', seg)
            cv2.waitKey()

