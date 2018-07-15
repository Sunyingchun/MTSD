import torchvision.models as models

from models.yolov3SPP import *

def get_model(name, n_classes, inputSize):
    model = _get_model_instance(name)

    if name == 'yolov3SPP':
        model = model(n_classes=n_classes, inputSize = inputSize)

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'yolov3SPP': yolov3SPP,
        }[name]
    except:
        print('Model {} not available'.format(name))
