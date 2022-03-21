from torch import nn
import torch
class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.loss = torch.nn.L1Loss()
    def forward(self,x,target):
        label = torch.zeros_like(x) + x
        c,w,h = label.shape
        for i in range(len(target)):
            boxes = (target[i]['boxes'] / 8).numpy().astype(int)
            area = 0
            for box in boxes:
                label[i,box[1]:box[3],box[0]:box[2]] = 0
                area = area + (box[2]-box[0])*(box[3]-box[1])

            tmp = label[i:i+1].clone()
            tmp = tmp.view(1,-1)
            res = torch.argsort(tmp,descending=True)
            tmp[0,res[0,0:area]] = 1
            tmp = tmp.view(1,w,h)
            label[i:i + 1] = tmp

        return (x-label).abs().mean(),x,label

if __name__=='__main__':
    loss = HeatmapLoss()
    target = [{'boxes': torch.Tensor([[246., 214., 286., 296.],
        [326., 234., 420., 306.],
        [276., 160., 328., 206.]]), 'labels': torch.Tensor([1, 1, 1])},{'boxes': torch.Tensor([[246., 214., 286., 296.],
        [326., 234., 420., 306.],
        [276., 160., 328., 206.]]), 'labels': torch.Tensor([1, 1, 1])}]
    x = torch.rand(2,60,80)

    l = loss(x,target)
    print(l)
