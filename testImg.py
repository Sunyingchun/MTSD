import os
import cv2
import torch
import argparse
import random
import config.config as config
import numpy as np
from PIL import Image
import scipy.misc as misc
import matplotlib.pyplot as plt

from models import get_model
from loader import cityscapesLoader
from torch.autograd import Variable
from tools.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, help='path to test picture or dictory')
parser.add_argument('--modelPath', type=str, help='path to model path')
parser.add_argument('--savePath', type=str, default='./result', help='path to save result')
opt = parser.parse_args()

def decode_segmap(pic, detections, nClass):
	colors = [[128,  64, 128],
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
	img = cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR) #convert PIL.image to cv2.mat
	
	r = detections.copy()
	g = detections.copy()
	b = detections.copy()
	for l in range(0, nClass):
		r[detections == l] = label_colours[l][0]
		g[detections == l] = label_colours[l][1]
		b[detections == l] = label_colours[l][2]

	rgb = np.zeros((detections.shape[0], detections.shape[1], 3))

	rgb[:, :, 0] = (r * 0.4 + img[:,:,2] * 0.6) / 255.0
	rgb[:, :, 1] = (g * 0.4 + img[:,:,1] * 0.6) / 255.0
	rgb[:, :, 2] = (b * 0.4 + img[:,:,0] * 0.6) / 255.0

	return rgb

def correct_yolo_boxes(rects, w, h, netw, neth, relative = 1):
	if(float(netw)/float(w) < float(neth)/float(h)):
		new_w=netw
		new_h=(h*netw)/w
	else:
		new_h=neth
		new_w=(w*neth)/h
	newRects = []
	for rect in rects:
		box_x = (rect[0] - (netw - new_w)/2./netw) / (float(new_w)/float(netw))
		box_y = (rect[1] - (neth - new_h)/2./neth) / (float(new_h)/float(neth))
		box_w = (rect[2] - rect[0])*float(netw)/float(new_w)
		box_h = (rect[3] - rect[1])*float(neth)/float(new_h)
		if relative:
			box_x = box_x*w
			box_w = box_w*w
			box_y = box_y*h
			box_h = box_h*h

		newRects.append([int(box_x), int(box_y), int(box_x + box_w), int(box_y + box_h), int(rect[4])])
	
	return newRects

def imageTransform(image):
	image = np.array(image, dtype = np.float32)
	h, w, _ = image.shape
	dim_diff = np.abs(h - w)
	# Upper (left) and lower (right) padding
	pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
	pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
	# Add padding
	in_ = np.pad(image, pad, 'constant', constant_values=128) / 255.
	padded_h, padded_w, _ = in_.shape
	# Resize and normalize
	in_ = cv2.resize(in_, (config.imgSize, config.imgSize))
	# Channels-first
	in_ = np.transpose(in_, (2, 0, 1))
	in_ = in_[np.newaxis, :]
	# As pytorch tensor
	in_ = torch.from_numpy(in_).float()

	return in_

def imageLoader(test):
	oriImgList = {}
	imgList = {}
	if os.path.isdir(test):
		for imgName in os.listdir(test):
			oriImage = Image.open(test+"/"+imgName)
			image = imageTransform(oriImage)
			oriImgList[imgName] = oriImage
			imgList[imgName] = image
	else:
		oriImage = Image.open(test)
		image = imageTransform(oriImage)
		imgName = test.split('/')[-1]
		oriImgList[imgName] = oriImage
		imgList[imgName] = image
	
	return oriImgList, imgList

def modelInit(modelPath):
	t_loader = cityscapesLoader(config.trainList, imgSize = config.imgSize)
	n_classes = t_loader.n_classes
	imgSize = t_loader.imgSize

	print("Loading Model from {} ......".format(modelPath))
	model = get_model(config.arch, n_classes, imgSize)
	state = convert_state_dict(torch.load(modelPath)['model_state'])
	#state = convert_state_dict(torch.load('./yolov3SPP_cityscapes.pth'))
	model.load_state_dict(state)
	model.cuda()

	model.eval()

	return model, n_classes

def modelForward(oriImgList, imgList, model, savePath, nClass, drawFig = True, saveFig = False):
	print("Forward Network ......")
	for imgName, img in imgList.items():
		print("Processing Image: {}".format(imgName))
		img = Variable(img.cuda())
		outputSeg, outputDet = model(img)

		# oriImage shape, input shape
		ori_w, ori_h = oriImgList[imgName].size
		pre_h, pre_w = config.imgSize, config.imgSize
		#detection
		detections = non_max_suppression(outputDet, config.numClasses, config.confThresh, config.iouThresh)[0]

		newRects = []
		if detections is not None:
			rects = []
			for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
				rects.append([x1 / pre_h, y1 / pre_h, x2 / pre_h, y2 / pre_h, cls_pred])
			newRects = correct_yolo_boxes(rects, ori_w, ori_h, pre_w, pre_h)
		
		#segmentation
		segmentations = np.squeeze(outputSeg.data.max(1)[1].cpu().numpy(), axis=0)
		pad = config.imgSize - int(float(config.imgSize) / float(max(ori_w, ori_h)) * float(min(ori_w, ori_h)))
		padW = [pad//2, pad - pad//2] if ori_h > ori_w else [0, 0]
		padH = [pad//2, pad - pad//2] if ori_w > ori_h else [0, 0]
		segmentations = segmentations[padH[0]:pre_h-padH[1], padW[0]:pre_w-padW[1]]
		segmentations = segmentations.astype(np.float32)
		segmentations = misc.imresize(segmentations, [ori_h, ori_w], 'nearest', mode='F') # float32 with F mode, resize back to orig_size
		decoded = decode_segmap(oriImgList[imgName], segmentations, nClass)
		
		Boxplot(decoded, newRects, imgName, drawFig, saveFig, savePath)
		if cv2.waitKey() & 0xFF == 27:
			break

def Boxplot(img, rects, imgName, drawFig, saveFig, savePath, scale = 0.6):
	className = config.className
	color_ink = dict(car = [0,255,0], bus = [255,255,0], person = [0,0,255], truck = [0,255,255],\
	                 bicycle = [255,0,0], motorbike = [255,0,255], trafficSign = [128,255,0])
	font = cv2.FONT_HERSHEY_SIMPLEX
	if drawFig:
		for rect in rects:
			cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color_ink[className[rect[4]]], 2)
			cv2.putText(img,str('%s' %className[rect[4]]),(rect[0],rect[1]-10),font,0.5,color_ink[className[rect[4]]],2)
		if saveFig:
			cv2.imwrite(savePath + '/' + imgName[:-4] + '_result.png', img * 255.0)
		cv2.namedWindow("image",0)
		cv2.resizeWindow("image", int(img.shape[1]*scale), int(img.shape[0]*scale))
		cv2.imshow('image', img)

def main():
	if not os.path.exists(opt.savePath):
		os.mkdir(opt.savePath)

	model, nClass = modelInit(opt.modelPath)
	oriImgList, imgList = imageLoader(opt.test)
	modelForward(oriImgList, imgList, model, opt.savePath, nClass, saveFig = True)

if __name__ == "__main__":
	main()

'''

'''
