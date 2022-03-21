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
vis = Visualizer('faster rcnn')
def eval(model,dataloader,device='cuda'):
    results = {}
    model.eval()
    cpu_device = torch.device("cpu")
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
        if i%100==0:
            print(i)
            image[0] = inverse_normalize(image[0])

            gt_img = visdom_bbox(image[0].cpu().numpy(),
                                 at.tonumpy(gt_bboxes_),
                                 at.tonumpy(gt_labels_))
            vis.img('gt_img', gt_img)

            pred_img = visdom_bbox(image[0].cpu().numpy(),
                                   at.tonumpy(pred_bboxes_),
                                   at.tonumpy(pred_labels_),
                                   at.tonumpy(pred_scores_))
            vis.img('pred_img', pred_img)


        predicts = []
        pred_bboxes_ = pred_bboxes_ / 2
        for fishI in range(len(pred_labels_)):
            tmp = ['fish', pred_scores_[fishI], pred_bboxes_[fishI].tolist()]
            predicts.append(tmp)

        for img_id, predict in zip([img_id[0]], [predicts]):
            results[img_id] = predict

    ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, r = dataloader.dataset.run_coco_eval(results, '.')


    print(r)

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

def get_optimizer(model):
    lr = 1e-3
    params = []
    for key, value in dict(model.named_parameters()).items():
         if value.requires_grad:
             if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
             else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 5e-4}]

    optimizer = torch.optim.SGD(params, momentum=0.9)
    return optimizer


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_tracking = opt.reid_dim > 0
    train_dataset = COCODataset(data_dir=opt.data_dir,
                                json_file=opt.train_ann,
                                img_size=opt.input_size,
                                tracking=do_tracking,
                                preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, tracking=do_tracking),
                                )
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=3,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=utils.collate_fn)

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
    # checkpoint = torch.load('save_weights_fish/vgg-model-2.pth')
    # model.load_state_dict(checkpoint['model'])


    # define optimizer
    optimizer = get_optimizer(model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    num_epochs = 10

    for epoch in range(num_epochs):
        print('classifer training ', epoch)

        utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=50, warmup=True)
        lr_scheduler.step()
        if epoch>=3:
            eval(model, val_data_loader,'cuda')


        # save weights
        if epoch > 1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights_fish/vgg-model-{}.pth".format(epoch))


if __name__ == "__main__":
    main()
