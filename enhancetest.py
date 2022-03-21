import torch.nn as nn
import torch
num = 21
conv = nn.Conv2d(1,1,num,stride=1,padding=10)
nn.init.uniform(conv.weight,1)
p_num = num*num
def enhance(img):
    img = torch.Tensor(img)
    img = img.permute(2,1,0)
    img = img.unsqueeze(dim=0)
    img = img/255
    mu = img.mean()

    mx_r = conv(img[:, 0:1,:,:])/p_num
    mx_g = conv(img[:, 1:2, :, :])/p_num
    mx_b = conv(img[:, 2:3, :, :])/p_num
    sigmax_r = conv((img[:,0:1,:,:] - mx_r)*(img[:,0:1,:,:] - mx_r)).sqrt()/p_num
    sigmax_g = conv((img[:, 1:2, :, :] - mx_g) * (img[:, 1:2, :, :] - mx_g)).sqrt()/p_num
    sigmax_b = conv((img[:, 2:3, :, :] - mx_b) * (img[:, 2:3, :, :] - mx_b)).sqrt()/p_num

    x_r = mx_r + mu/sigmax_r * (img[0,0:1,:,:] - mx_r)
    x_g = mx_g + mu/sigmax_g * (img[0,1:2,:,:] - mx_g)
    x_b = mx_b + mu/sigmax_b * (img[0,2:3,:,:] - mx_b)
    print(x_r)
    x = torch.cat([x_r,x_g,x_b],dim=1)
    x = x.squeeze(dim=0)
    return x

import cv2

img = cv2.imread('7772.jpg')
img = enhance(img)
img = img.detach().numpy().transpose((1,2,0))
cv2.imshow('0',img*255)
cv2.waitKey(0)