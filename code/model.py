import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from Densenet import DenseNet
def myOwnLoad(model, check):
    modelState = model.state_dict()
    tempState = OrderedDict()
    print(len(check.keys()))
    print(len(model.state_dict()))
    for i in range(len(modelState .keys())):

        tempState[list(modelState.keys())[i]] = check[list(check.keys())[i]]
    #temp = [[0.02]*1024 for i in range(200)]  # mean=0, std=0.02
    #tempState['myFc.weight'] = torch.normal(mean=0, std=torch.FloatTensor(temp)).cuda()
    #tempState['myFc.bias']   = torch.normal(mean=0, std=torch.FloatTensor([0]*200)).cuda()
    model.load_state_dict(tempState)
    return model

class SiameseNet(nn.Module):
    def __init__(self, feat_dim=50,num_classes=2):
        super(SiameseNet, self).__init__()
        self.basenet=DenseNet()
        model_weight = torch.load("checkpoints/exp1/ model_10.pkl")['weight']
        self.basenet = myOwnLoad(self.basenet, model_weight)
        #self.basenet.load_state_dict(torch.load("checkpoints/exp1/ model_10.pkl")['weight'])
        self.classifier1 = nn.Linear(684, feat_dim)
        self.classifier2 = nn.Linear(feat_dim, num_classes)
        self.classifier = nn.Linear(50,num_classes)
    def forward(self, x,y):
        f_x=self.basenet(x)
        f_y=self.basenet(y)
        #f = torch.cat((f_x,f_y),1)
        #print(f.size())
        f = f_x - f_y
        #out1 = self.classifier1(f)
        #out2 = F.sigmoid(self.classifier2(out1))
        #out2 = self.classifier2(out1)
        out = self.classifier(f)

        return out


