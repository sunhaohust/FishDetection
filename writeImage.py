import torch
import torchvision

from data.coco import COCODataset
from data.data_augment import TrainTransform
from train_utils.train_eval_utils import inverse_normalize
from utils.vis_tool import Visualizer
from utils.vis_tool import visdom_bbox
from config import opt
from network_files.faster_rcnn_framework import FasterRCNN
from network_files.rpn_function import AnchorsGenerator
from torchvision.models import vgg16
from torchvision.models import AlexNet
from torchvision.models import resnet101
from torchvision.ops import misc
from utils.vis_tool import visdom_bbox
from train_utils import train_eval_utils as utils
import os
from utils.eval_tool import eval_detection_voc
from utils import array_tool as at
from torch import nn
import cv2
import numpy as np
vis = Visualizer('faster rcnn')
def writeImg(img,path=None):
    img = img.transpose(1, 2, 0).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img * 255)
def eval(dataloader):
    for i, (images, targets, img_info, img_id,img_bgs) in enumerate(dataloader):

        path = 'pre_img/image/' + str(img_id[0].numpy()) + '.jpg'
        image = images[0].numpy()
        cv2.imwrite(path, image)

        path = 'pre_img/background/' + str(img_id[0].numpy()) + '.jpg'
        bg = img_bgs[0].numpy()
        cv2.imwrite(path, bg)




def main():


    val_dataset = COCODataset(
        data_dir=opt.data_dir,
        json_file=opt.val_ann,
        name="val2014",
        img_size=opt.test_size,
        tracking=False,
        preproc=None)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=4)


    eval(val_data_loader)

if __name__ == "__main__":
    main()
