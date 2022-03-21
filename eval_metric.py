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
def eval(model,dataloader,device='cuda'):
    results = {}
    model.eval()
    cpu_device = torch.device("cpu")
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i, (images, targets, img_info, img_id,img_bgs) in enumerate(dataloader):
        image = torch.tensor(images).to(device)
        img_bg = torch.tensor(img_bgs).to(device)
        targets = targets[0]
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        with torch.no_grad():
            outputs = model(image,img_bg)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs][0]

        gt_bboxes_ = targets['boxes']
        gt_labels_ = targets['labels']

        pred_bboxes_ = outputs['boxes']
        pred_labels_ = outputs['labels']
        pred_scores_ = outputs['scores']
        gt_bboxes += [gt_bboxes_.to('cpu').numpy().copy()]
        gt_labels += [gt_labels_.to('cpu').numpy().copy()]

        pred_bboxes += [pred_bboxes_.numpy().copy()]
        pred_labels += [pred_labels_.numpy().copy()]
        pred_scores += [pred_scores_.numpy().copy()]
        if (i+1)%100==0:
            print(i)

    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, None,
                                use_07_metric=False)

    print(result)




def create_model(num_classes):
    model = resnet101(pretrained=True)
    backbone =nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3
    )
    backbone.out_channels = 1024
    anchor_generator = AnchorsGenerator(sizes=((64,128,256,512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_tracking = opt.reid_dim > 0

    val_dataset = COCODataset(
        data_dir=opt.data_dir,
        json_file=opt.val_ann,
        name="val2014",
        img_size=opt.test_size,
        tracking=do_tracking,
        preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120, tracking=do_tracking,
                               augment=False))
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      collate_fn=utils.collate_fn)

    model = create_model(num_classes=2)
    model.to(device)
    checkpoint = torch.load('save_weights_fish/best.pth')

    model.load_state_dict(checkpoint['model'])


    eval(model, val_data_loader,'cuda')

if __name__ == "__main__":
    main()
