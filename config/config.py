# select architecture
arch = "yolov3SPP"

# pretrained model
pretrainedModel = "yolov3SPP_cityscapes.pth"

# resume from checkpoint
resume = "./snapshot/yolov3SPP_best_model.pkl"

# learning rate
base_lr = 0.001
lr_decay_epochs = [120,160]
lr_decay = 0.1

max_epoch = 200

# L2 regularizer
weight_decay = 0.0005
momentum = 0.9

snapshot = 2
snapshot_path = "snapshot"

# dataSet
#imdb_train = "/home/lipj/Car_identify/data/DayData/train/"
#imdb_val = "/home/lipj/Car_identify/data/DayData/val/"
imgSize = 608
trainList = "/home/wfw/data/VOCdevkit/cityscapesMTSD/train.txt"
valList = "/home/wfw/data/VOCdevkit/cityscapesMTSD/val.txt"
train_batch_size = 8
test_batch_size = 1

# yolo param
anchors = [[[58,135], [150,122], [208,203]],
           [[26,64], [33,26], [61,44]],
           [[8,9], [12,29], [18,15]]]
className = ['truck', 'person', 'bicycle', 'car', 'motorbike', 'bus']
numClasses = 6
iouThresh = 0.45
confThresh=0.24

jitter = 0.3
hue = 0.1
saturation = 1.5
exposure = 1.5

# GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.

# SETTINGS FOR DISPLAYING ON TENSORBOARD
visdomTrain = False
visdomVal = False
