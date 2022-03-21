import math
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2, padding=5),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=2))
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        #         nn.init.normal_(m.weight,0,2)
    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)

        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4)

        return conv5

if __name__=='__main__':
    img = torch.Tensor(1,3,480,640)
    model = AlexNet()
    print(model(img).shape)