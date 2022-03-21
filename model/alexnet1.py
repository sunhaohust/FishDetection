import math
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        ch1 = 96
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=1, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=2, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=2, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(3, 96, 21, stride=2, padding=10),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True))


    def forward(self, x):
        conv1 = self.conv1(x)


        return conv1

if __name__=='__main__':
    img = torch.Tensor(1,3,480,640)
    model = AlexNet()
    print(model(img).shape)