import timm 
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.fc = nn.Sequential(nn.Linear(1, 2))
    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x_dist = (x1 - x2).pow(2).sum(1).sqrt()
        x_dist = torch.unsqueeze(x_dist, dim=1)
        x = self.fc(x_dist)
        return x

class TripletLossModel(nn.Module):
    def __init__(self):
        super(TripletLossModel, self).__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True)
    def forward(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)
        return x1, x2, x3

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True)
        self.fc = nn.Sequential(nn.Linear(1000, 24480))
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ClassifierArcface(nn.Module):
    def __init__(self):
        super(ClassifierArcface, self).__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True)
        self.fc = ArcFace(1000, 24480)
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=30, m=0.3, stride=0.1, max_m=0.8):
        super().__init__()
        self.m = m
        self.s = s
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.cos_m = torch.cos(torch.tensor(self.m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)
        self.last_epoch = 0
        self.max_m = max_m
        self.m_s = stride

    def update(self, c_epoch):
        self.m = min(self.m + self.m_s * (c_epoch - self.last_epoch), self.max_m)
        self.last_epoch = c_epoch
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.cos_m = torch.cos(torch.tensor(self.m))

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)
        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos**2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
        return cos

def main():
    model = SiameseNet()
    model.cuda()
    x = torch.randn(5, 3, 224, 224)
    x = x.cuda()
    print(model(x, x).shape)


if __name__ == "__main__":
    main()