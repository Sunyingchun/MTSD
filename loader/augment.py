# encoding: utf-8
import random
import os
import cv2
from PIL import Image
import numpy as np

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, segMap, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    croppedSeg = segMap.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)
    sizedSeg = croppedSeg.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        sizedSeg = sizedSeg.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, sizedSeg, flip, dx,dy,sx,sy 

def fill_truth_detection(lbl, w, h, flip, dx, dy, sx, sy):
    cc = 0
    for i in range(lbl.shape[0]):
        x1 = lbl[i][1] - lbl[i][3]/2
        y1 = lbl[i][2] - lbl[i][4]/2
        x2 = lbl[i][1] + lbl[i][3]/2
        y2 = lbl[i][2] + lbl[i][4]/2
        
        x1 = min(0.999, max(0, x1 * sx - dx)) 
        y1 = min(0.999, max(0, y1 * sy - dy)) 
        x2 = min(0.999, max(0, x2 * sx - dx))
        y2 = min(0.999, max(0, y2 * sy - dy))
        
        lbl[i][1] = (x1 + x2)/2
        lbl[i][2] = (y1 + y2)/2
        lbl[i][3] = (x2 - x1)
        lbl[i][4] = (y2 - y1)

        if flip:
            lbl[i][1] =  0.999 - lbl[i][1] 
        
        if lbl[i][3] < 0.001 or lbl[i][4] < 0.001:
            continue

        cc += 1
        if cc >= 50:
            break

    return lbl

def augmentation(img, labels, segMap, shape, jitter, hue, saturation, exposure):
    ## data augmentation
    img, segMap, flip, dx, dy, sx, sy = data_augmentation(img, segMap, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labels, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)

    return img, label, segMap
