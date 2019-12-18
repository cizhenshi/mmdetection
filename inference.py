import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.models import build_detector
import glob, os, json
import numpy as np

config_file = 'configs/da/faster_vgg16.py' #config file
#checkpoint_file = '../data/convert.pth' # checkpoint file
checkpoint_file = './work_dirs/city_base/epoch_12.pth' # checkpoint file
# build the model from a config file and a checkpoint file
print('loading model...')
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print('loading complete!')
img = "../data/test.png"
res = inference_detector(model, img)
classnames = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']
show_result(img, res, classnames, score_thr=0, show=False, out_file="./det.jpg")