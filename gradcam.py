import argparse
import cv2
import numpy as np
import torchvision
import torch

from model.resnet101 import ResNet101
from network_files.rpn_function import AnchorsGenerator
from torch.autograd import Function
import torch.nn.functional as F
from torchvision.models import resnet101
import torch.nn as nn
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor



class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x,x_bg):
        outputs = []
        self.gradients = []
        x = x.squeeze(0)
        target_index = None
        x, target_index = self.model.transform.resize(x,target_index)
        x = x.unsqueeze(0)
        x = x.to('cuda')
        x.requires_grad_(True)

        x_bg = x_bg.squeeze(0)
        target_index = None
        x_bg, target_index = self.model.transform.resize(x_bg, target_index)
        x_bg = x_bg.unsqueeze(0)
        x_bg = x_bg.to('cuda')
        x_bg.requires_grad_(True)

        heatmap_0, heatmap_8 = self.model.bgnet(x, x_bg)
        heatmap_8 = torch.unsqueeze(heatmap_8, dim=1)
        heatmap_8 = torch.nn.functional.interpolate(heatmap_8, scale_factor=16, mode='nearest',
                                                    align_corners=None)

        heatmap_8 = torch.repeat_interleave(heatmap_8, 3, dim=1)
        x = x * heatmap_8
        x = self.model.backbone(x)

        x.requires_grad_(True)
        x.register_hook(self.save_gradient)
        outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x,x_bg):

        target_activations, x = self.feature_extractor(x,x_bg)

        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)

        # print("这里应该有梯度")
        # print(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)

    return input


def show_cam_on_image(img, mask,filename):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))

    cam = heatmap + np.float32(img)
    cam = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
    cv2.imwrite(filename, np.uint8(255 * cam))



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, input_bg,index=None):

        features, output = self.extractor(input.cuda(),input_bg.cuda())

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)

        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()


        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)

        _,_,w,h = input.shape
        cam = cv2.resize(cam, (h,w))

        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # if self.cuda:
        #     output = self.forward(input.cuda())
        # else:
        #     output = self.forward(input)
        # print("这里多余requires_grad=True")
        if self.cuda:
            input = input.cuda()

        input = input.requires_grad_(True)

        # print("输入的梯度项为requires_grad=True")
        x = input.squeeze(0)
        # print(x.shape)
        target_index = None
        x, target_index = self.model.transform.resize(x, target_index)
        # print(x.shape)
        x = x.unsqueeze(0)
        # print("这里没梯度")
        x.requires_grad_(True)
        # print(x)
        x = self.model.backbone.conv1(x)
        # print(x.shape)
        x = self.model.backbone.bn1(x)
        # print(x.shape)
        x = self.model.backbone.relu(x)
        # print(x)
        x = self.model.backbone.maxpool(x)
        # print("这里没有梯度项")
        x = self.model.backbone.layer1(x)
        x1 = x

        x = self.model.backbone.layer2(x)
        x2 = x

        x = self.model.backbone.layer3(x)
        x3 = x


        # x1 = nn.Conv2d(256, 256, 1)(x1)
        # # print(x1.shape)
        # x1 = x1 + F.interpolate(x1, size=[200,200], mode="nearest")
        # # print(x1.shape)
        # x1 = nn.Conv2d(256, 256, 3, padding=1)(x1)
        # # print(x1.shape)
        # x2 = nn.Conv2d(512, 256, 1)(x2)
        # # print(x2.shape)
        # x_2 = F.interpolate(x1, size=[100, 100], mode="nearest")
        # # x_2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # # print(x_2.shape)
        # x2 = x2 + x_2
        # # print(x2.shape)
        # x2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x2 = nn.Conv2d(256, 256, 3, padding=1)(x2)
        # # print(x2.shape)
        # x3 = nn.Conv2d(1024, 256, 1)(x3)
        # x3 = x3 + F.interpolate(x1, size=[50,50], mode="nearest")
        # x3 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x3 = nn.Conv2d(256, 256, 3, padding=1)(x3)
        # x4 = nn.Conv2d(2048, 256, 1)(x4)
        # x4 = x4 + F.interpolate(x1, size=[25,25], mode="nearest")
        # x4 =  F.interpolate(x1, size=[200, 200], mode="nearest")
        # x4 = nn.Conv2d(256, 256, 3, padding=1)(x4)

        # x = x4 + x3 + x2 + x1
        x = x3
        # x = self.avgpool(x)
        x = self.maxpool(x)
        output = x.view(x.size(0), -1)

        # output = self.forward(input)
        #
        # print(output.shape)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.cpu().detach().numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./test.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


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
    model = model.to('cuda')
    print(model)
    return model

def writeGradcam(imgname,bgname,grad_cam):
    img = cv2.imread(imgname)
    img = np.float32(cv2.resize(img, (640, 480))) / 255
    input = preprocess_image(img)

    img_bg = cv2.imread(bgname)
    img_bg = np.float32(cv2.resize(img_bg, (640, 480))) / 255
    input_bg = preprocess_image(img_bg)
    target_index = None
    mask = grad_cam(input, input_bg,target_index)
    filename = imgname.replace('image','gradcam')
    print(filename)
    show_cam_on_image(img, mask,filename)

if __name__ == '__main__':

    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2)
    # print(model)
    train_weights = "./save_weights_fish/best.pth"
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"], False)
    model.to(device)

    grad_cam = GradCam(model=model, feature_module=model.backbone, \
                        target_layer_names=["0"], use_cuda=args.use_cuda)

    filename = ['15','358','805','3749','4647','4673','4764','4774','5504','5721','6348','6370','6587','7601','7672']
    for name in filename:
        path = 'pre_img/image/' + name + '.jpg'
        bgname = 'pre_img/background/' + name + '.jpg'
        writeGradcam(path, bgname, grad_cam)


