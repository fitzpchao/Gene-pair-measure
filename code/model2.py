import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Densenet2 import DenseNet

class SiameseNet(nn.Module):
    def __init__(self, feat_dim=50,num_classes=2):
        super(SiameseNet, self).__init__()
        self.basenet=DenseNet()
        self.classifier1 = nn.Linear(684, feat_dim)
        self.classifier2 = nn.Linear(feat_dim, num_classes)
    def forward(self, x,y):
        f_x=self.basenet(x)
        f_y=self.basenet(y)
        f = torch.cat([f_x,f_y],1)
        #f = f_x - f_y
        out1 = self.classifier1(f)
        #out2 = F.sigmoid(self.classifier2(out1))
        #out2 = self.classifier2(out1)

        return out1


