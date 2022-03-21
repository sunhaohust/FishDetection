import torch
from torch import nn
from collections import OrderedDict
from network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
from network_files.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):

    def __init__(self, backbone,bgnet, rpn,roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.bgnet = bgnet
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, img_bgs, targets=None):
        # type: (List[Tensor], List[Tensor],Optional[List[Dict[str, Tensor]]])
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)  # 对图像进行预处理
        img_bgs, _ = self.transform(img_bgs)
        heatmap_0, heatmap_8 = self.bgnet(images.tensors, img_bgs.tensors)
        heatmap_8 = torch.unsqueeze(heatmap_8, dim=1)
        heatmap_8 = torch.nn.functional.interpolate(heatmap_8, scale_factor=16, mode='nearest',
                                                    align_corners=None)

        heatmap_8 = torch.repeat_interleave(heatmap_8, 3, dim=1)
        images.tensors = images.tensors * heatmap_8
        # enhance last feature
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        # heatmap = torch.unsqueeze(heatmap_8,dim=1)
        # heatmap = torch.repeat_interleave(heatmap,1024,dim=1)
        # features_rpn = features*heatmap

        # #
        # if isinstance(features_rpn, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
        #     features_rpn = OrderedDict([('0', features_rpn)])  # 若在多层特征层上预测，传入的就是一个有序字典
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典



        # 将特征层以及标注target信息传入rpn中
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.training:
            detections_1, detector_losses_1 = self.roi_heads[0](features, proposals, images.image_sizes, targets)
            detections_2, detector_losses_2 = self.roi_heads[1](features, detections_1, images.image_sizes, targets)
            detections_3, detector_losses_3 = self.roi_heads[2](features, detections_2, images.image_sizes, targets)
        else:
            detections_1, detector_losses_1 = self.roi_heads[0](features, proposals, images.image_sizes, targets)
            detections_2, detector_losses_2 = self.roi_heads[1](features, detections_1[0]['cascade_proposals'],
                                                                images.image_sizes, targets)
            detections_3, detector_losses_3 = self.roi_heads[2](features, detections_2[0]['cascade_proposals'],
                                                                images.image_sizes, targets)

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分

        if not self.training:
            class_logits = (detections_1[0]['class_logits'] + \
                            detections_2[0]['class_logits'] + \
                            detections_3[0]['class_logits']) / 3
            box_regression = detections_3[0]['box_regression']
            proposals = detections_3[0]['proposals']
            boxes, scores, labels = self.roi_heads[2].postprocess_detections(class_logits, box_regression, proposals,
                                                                             images.image_sizes)
            num_images = len(boxes)
            detections = []
            for i in range(num_images):
                detections.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        for k in detector_losses_1.keys():
            detector_losses_1[k] += detector_losses_2[k] * 0.5 + detector_losses_3[k] * 0.25
        losses.update(detector_losses_1)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections




class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()

        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()
        self.conv_8 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=2, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8, padding=0))
        for param in self.conv_8.parameters():
            param.requires_grad = False

        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=1, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))
        for param in self.conv_0.parameters():
            param.requires_grad = False
    def forward(self, x, x_bg):
        x_8 =self.conv_8(x)
        x_bg_8 = self.conv_8(x_bg)
        h_8 = self.heatmap(x_8, x_bg_8)
        h_8 = 1 - (h_8 - h_8.min()) / (h_8.max() - h_8.min())

        x_0 = self.conv_0(x)
        x_bg_0 = self.conv_0(x_bg)
        h_0 = self.heatmap(x_0, x_bg_0)
        h_0 = 1 - (h_0 - h_0.min()) / (h_0.max() - h_0.min())
        return h_0,h_8


    def heatmap(self,x, x_bg):
        imgs_norm = torch.sqrt(torch.sum(x * x, dim=1))
        img_bgs_norm = torch.sqrt(torch.sum(x_bg * x_bg, dim=1))
        out = torch.sum(x * x_bg, dim=1)
        heatmap = out / (imgs_norm * img_bgs_norm)
        return heatmap
class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=480, max_size=640,      # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,    # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.3, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,   # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None,cascade_iou_thr = [0.5, 0.6, 0.7]):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        bgnet = BGNet()
        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        cascade_bbox_reg_weights = [(10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.)]
        roi_heads = nn.ModuleList()
        for i in range(3):
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

            box_fg_iou_thresh = box_bg_iou_thresh = cascade_iou_thr[i]
            bbox_reg_weights = cascade_bbox_reg_weights[i]
            roi_head = RoIHeads(
                # Box
                box_roi_pool, box_head, box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,
                box_batch_size_per_image, box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh, box_nms_thresh, box_detections_per_img)

            roi_heads.append(roi_head)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # imagenet
            # image_mean = [0.4336, 0.5004, 0.4596]  # fish
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]   # imagenet
            # image_std = [0.236, 0.315, 0.313]   # fish

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, bgnet,rpn,roi_heads, transform)
