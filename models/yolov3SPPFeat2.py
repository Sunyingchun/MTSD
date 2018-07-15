import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

import sys
sys.path.append("/home/wfw/wissen_work/PytorchProject/Pytorch-MTSD")

from tools import caffe_pb2
from models.utils import *
from tools.loss import *
import config.config as config

def _make_layers(net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer = _make_layers(sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            convType, in_channels, n_filters = item
            if convType == 'N':
                layers.append(conv2DBatchNormRelu(in_channels, n_filters, k_size=3, stride=1, 
                                                  padding=1, bias=False, leaky_relu=True))
            elif convType == 'B':
                layers.append(residualBlockYolo(in_channels, n_filters))
            elif convType == 'E':
                layers.append(residualBlockYoloExp(in_channels, n_filters))
            elif convType == 'nE':
                layers.append(BlockYoloExp(in_channels, n_filters))

    return nn.Sequential(*layers)

class yolov3SPP(nn.Module):
    
    """
    Multi-tast Semantic Segmentation and Object Detection Network

    """

    def __init__(self, 
                 n_classes=21,  
                 inputSize=608):

        super(yolov3SPP, self).__init__()

        self.n_classes = n_classes
        self.input_size = (inputSize, inputSize)

        net_cfgs = [
            # conv1s
            [('N', 3, 32)],
            [('B', 32, 64)],
            [('B', 64, 128), ('E', 128, 64)],
            [('B', 128, 256), ('E', 256, 128), ('E', 256, 128), ('E', 256, 128), ('E', 256, 128),\
             ('E', 256, 128), ('E', 256, 128), ('E', 256, 128)],
            # conv2
            [('B', 256, 512), ('E', 512, 256), ('E', 512, 256), ('E', 512, 256), ('E', 512, 256),\
             ('E', 512, 256), ('E', 512, 256), ('E', 512, 256)],
            # conv3
            [('B', 512, 1024), ('E', 1024, 512), ('E', 1024, 512), ('E', 1024, 512)],
            # conv4
            [('nE', 1024, 512), ('nE', 1024, 512)],
            # conv5
            [('nE', 768, 256), ('nE', 512, 256)],
            # conv6
            [('nE', 384, 128), ('nE', 256, 128), ('nE', 256, 128)],
        ]        

        # darknet
        self.conv1s = _make_layers(net_cfgs[0:4])
        self.conv2 = _make_layers(net_cfgs[4])
        self.conv3 = _make_layers(net_cfgs[5])
        # ---
        self.conv4 = _make_layers(net_cfgs[6])

        self.conv27_1 = conv2DBatchNormRelu(1024, 512, k_size=1, stride=1, 
                                                  padding=0, bias=False, leaky_relu=True)
        self.conv27_2 = conv2DBatchNormRelu(512, 1024, k_size=3, stride=1, 
                                                  padding=1, bias=False, leaky_relu=True)

        ####################YOLO#######################
        # Pyramid feature1
        self.conv28 = nn.Conv2d(1024, 33, kernel_size=1,
                                 padding=0, stride=1)

        # Pyramid feature2
        self.conv29 = conv2DBatchNormRelu(512, 256, k_size=1, stride=1, 
                                                  padding=0, bias=False, leaky_relu=True)

        self.conv5 = _make_layers(net_cfgs[7])
        self.conv32_1 = conv2DBatchNormRelu(512, 256, k_size=1, stride=1, 
                                                  padding=0, bias=False, leaky_relu=True)
        self.conv32_2 = conv2DBatchNormRelu(256, 512, k_size=3, stride=1, 
                                                  padding=1, bias=False, leaky_relu=True)
        self.conv33 = nn.Conv2d(512, 33, kernel_size=1,
                                 padding=0, stride=1)

        # Pyramid feature3
        self.conv34 = conv2DBatchNormRelu(256, 128, k_size=1, stride=1, 
                                                  padding=0, bias=False, leaky_relu=True)

        self.conv6 = _make_layers(net_cfgs[8])
        self.conv38 = nn.Conv2d(256, 33, kernel_size=1,
                                 padding=0, stride=1)

        self.YOLOLossFeat0 = YOLOLoss(config.anchors[0], config.numClasses, config.imgSize)
        self.YOLOLossFeat1 = YOLOLoss(config.anchors[1], config.numClasses, config.imgSize)
        self.YOLOLossFeat2 = YOLOLoss(config.anchors[2], config.numClasses, config.imgSize)

        # freeze the param
        #for param in self.parameters():
        #    param.requires_grad=False

        # finetune the param
        self.finetune_params = []
        for param in self.parameters():
            self.finetune_params.append(param)

        ##################PSP#########################
        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(512, [6, 3, 2, 1])

        # concat+1*1 conv
        self.concatConv = conv2DBatchNorm(1024, 512,  k_size=1, stride=1,
                                                  padding=0)
        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, 
                                                  padding=1, bias=False, leaky_relu=True)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)
        self.loss_fn = cross_entropy2d

    def forward(self, x, labels=None, segMaps=None):
        inp_shape = x.shape[2:]

        conv1s = self.conv1s(x)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        ###############################################
        convfinal1 = self.conv27_2(self.conv27_1(conv4))

        upsample1 = F.upsample(self.conv29(self.conv27_1(conv4)), conv2.shape[2:], mode='bilinear')
        cat1 = torch.cat([upsample1, conv2], 1)
        convfinal2 = self.conv32_2(self.conv32_1(self.conv5(cat1)))

        upsample2 = F.upsample(self.conv34(self.conv32_1(self.conv5(cat1))), conv1s.shape[2:], mode='bilinear')
        cat2 = torch.cat([upsample2, conv1s], 1)
        convfinal3 = self.conv6(cat2)
        ##################PSP#########################
        pda = self.pyramid_pooling(convfinal2)
        pda = self.concatConv(pda)

        final = self.cbr_final(pda)
        final = self.dropout(final)

        score = self.classification(final)
        upsample = F.upsample(score, size=inp_shape, mode='bilinear')

        ####################YOLO#######################
        feat1 = self.conv28(convfinal1)
        feat2 = self.conv33(convfinal2)
        feat3 = self.conv38(convfinal3)

        if labels is not None and segMaps is not None:
            loss_seg = self.loss_fn(upsample, segMaps)

            losses = []
            _loss_items = []
            _loss_items.append(self.YOLOLossFeat0(feat1, labels))
            _loss_items.append(self.YOLOLossFeat1(feat2, labels))
            _loss_items.append(self.YOLOLossFeat2(feat3, labels))
            for _loss_item in _loss_items:
                for j, l in enumerate(_loss_item):
                    if j == 0:
                        losses.append(l)
            loss_det = sum(losses)

            return loss_seg, loss_det

        else:
            objects = []
            objects.append(self.YOLOLossFeat0(feat1))
            objects.append(self.YOLOLossFeat1(feat2))
            objects.append(self.YOLOLossFeat2(feat3))
            output = torch.cat(objects, 1)
        
        return upsample, output

# For Testing Purposes only
if __name__ == '__main__':
    #from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl
    yolov3SPP = yolov3SPP(19, 608)
    x = torch.ones(2,3,608,608)
    x = Variable(x.type(torch.FloatTensor))
    y = torch.ones(2,608,608)
    y = Variable(y.type(torch.LongTensor))
    labels = torch.ones((2, 50, 5)).type(torch.FloatTensor) * 0.5
    labels = Variable(labels, requires_grad=False)
    up1, objects = yolov3SPP(x, labels, y)
    print(up1, objects)
