import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import config.config as config
from models import get_model
from loader import cityscapesLoader
from tools.metrics import runningScore
from tools.augmentations import *
from tools.utils import *

import warnings
warnings.filterwarnings("ignore")

def train():
    # Initial
    # mkdir snapshotPath
    if not os.path.exists(config.snapshot_path):
       os.mkdir(config.snapshot_path)

    # Setup Dataloader
    t_loader = cityscapesLoader(config.trainList, split = 'train', batchSize = config.train_batch_size, imgSize = config.imgSize, is_augmentation = False, randomResize = False)
    v_loader = cityscapesLoader(config.valList, split = 'val', imgSize = config.imgSize)

    n_classes = t_loader.n_classes
    imgSize = t_loader.imgSize
    trainloader = data.DataLoader(t_loader, batch_size=config.train_batch_size, num_workers=8)#not shuffle here, it will break because diffient shape
    valloader = data.DataLoader(v_loader, batch_size=config.test_batch_size, num_workers=8)

    # Setup Metrics for Iou calculate
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = get_model(config.arch, n_classes, imgSize)
    finetune_params = model.finetune_params
    #model = yolov3SPP(version='cityscapes', n_classes=19)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        # freeze the param
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        # finetune the param
        train_params = []
        for idx , param in enumerate(model.parameters()):
            if idx > len(finetune_params):
                train_params.append(param)
        optimizer = torch.optim.SGD([{'params': finetune_params}, {'params': train_params, 'lr': config.base_lr * 1}],\
                                     lr=config.base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        #for param_group in optimizer.param_groups:
        #    print("{} : {}".format(param_group['params'], param_group['lr']))
        # nomal optimizer
        #optimizer = torch.optim.SGD(model.parameters(), lr=config.base_lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # learning method
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs, gamma=config.lr_decay)

    if config.resume is not None:                                         
        if os.path.isfile(config.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(config.resume)) 
    else:
        #load_pretrained_model
        print("Loading pretrained Model: {}".format(config.pretrainedModel))
        start_epoch = 0 
        if config.pretrainedModel.split(".")[-1] == "pth":
            model.load_state_dict(torch.load(config.pretrainedModel))
        else:
            model.load_state_dict(torch.load(config.pretrainedModel)['model_state'])

    # initial visdom
    if config.visdomTrain:
        fig=plt.figure()
        # train visdom
        ax1 = fig.add_subplot(2,1,1)
        ax1.axis([start_epoch * len(trainloader), (start_epoch+1) * len(trainloader), 0, 1])
        ax1.plot(-1, -1, 'bo', label = 'LossSeg')
        ax1.plot(-1, -1, 'r^', label = 'lossDet')
        ax1.legend(loc='upper left')
        plt.title('LossSeg vs lossDet')
        # val visdom
        ax2 = fig.add_subplot(2,1,2)
        ax2.axis([start_epoch * len(trainloader), (start_epoch+1) * len(trainloader), 0, 1])
        ax2.plot(-1, -1, 'cs', label = 'LossSegVal')
        ax2.plot(-1, -1, 'y*', label = 'lossDetVal')
        ax2.legend(loc='upper left')

    bestIou = -100.0 
    bestmAP = -100.0
    lossSegDict = {}
    lossDetDict = {}
    for epoch in range(start_epoch, config.max_epoch):
        # update axis for visdom
        if config.visdomTrain:
            ax1.axis([start_epoch * len(trainloader), (epoch+1) * len(trainloader), 0, 1])
            ax2.axis([start_epoch * len(trainloader), (epoch+1) * len(trainloader), 0, 1])

        # model train pocess
        model.train()
        for i, (images, labels, segMaps) in enumerate(trainloader): 
            currentIter = epoch * len(trainloader) + i     
            poly_lr_scheduler(optimizer, config.base_lr, currentIter, max_iter = config.max_epoch * len(trainloader))

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            segMaps = Variable(segMaps.cuda())

            optimizer.zero_grad()
            loss_seg, loss_det = model(images, labels, segMaps)

            # fuse loss
            # loss = loss_seg + loss_det

            loss_seg.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                if config.visdomTrain:
                    lossSegDict[currentIter] = loss_seg.data[0]
                    lossDetDict[currentIter] = loss_det.data[0]
                    for perEpoch, lossSeg in lossSegDict.items():

                        ax1.plot(perEpoch, lossSeg, 'bo', label = 'LossSeg')
                        ax1.plot(perEpoch, lossDetDict[perEpoch], 'r^', label = 'lossDet')
                        plt.pause(0.033)

                print("[Epoch %d/%d, Batch %d/%d] Learning_rate: %.7f Loss_seg: %.4f Loss_det: %.4f" % \
                (epoch+1, config.max_epoch, i, len(trainloader), optimizer.param_groups[0]['lr'], loss_seg.data[0], loss_det.data[0]))

        # model eval pocess
        lossSegVal = []
        lossDetVal = []
        model.eval()
        APs = []
        for i_val, (images_val, labels_val, segMap_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)
            segMap_val = Variable(segMap_val.cuda(), volatile=True)

            outputSeg, outputDet = model(images_val)
            if config.visdomVal:            
                loss_segVal, loss_detVal = model(images_val, labels_val, segMap_val)   
                lossSegVal.append(loss_segVal.data[0]) 
                lossDetVal.append(loss_detVal.data[0])

            pred = outputSeg.data.max(1)[1].cpu().numpy()
            gt = segMap_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

            AP = evalDet(outputDet, labels_val, config.numClasses, config.imgSize, config.confThresh, config.iouThresh)
            APs.append(AP)   

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        print("Mean Average Precision: %.4f" % np.mean(APs))

        if config.visdomVal:
            ax2.plot((epoch+1) * len(trainloader), np.mean(lossSegVal), 'cs', label = 'LossSegVal')
            ax2.plot((epoch+1) * len(trainloader), np.mean(lossDetVal), 'y*', label = 'lossDetVal')
            plt.pause(0.033)  

        # write result to log
        with open('MTSD.log', 'a') as f:
            f.write("++++++++++MTSD Result+++++++++++++\nepoch: {} \nDetection result: \nMean Iou: {} \nSegmentation result: \nmAP: {}\n".\
                     format(epoch+1, score['Mean IoU : \t'], np.mean(APs)))

        if score['Mean IoU : \t'] >= bestIou:# or np.mean(APs) > bestmAp:
            bestIou = score['Mean IoU : \t']
            bestmAp = np.mean(APs)
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}/{}_best_model.pkl".format(config.snapshot_path, config.arch))

if __name__ == '__main__':
    train()

