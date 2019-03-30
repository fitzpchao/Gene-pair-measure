import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import random
from osgeo import gdalnumeric
import os

def load_img(path):
    img = gdalnumeric.LoadFile(path)
    #img = np.transpose(img,[1,2,0])
    img = np.array(img, dtype="float")
    '''B, G, R = cv2.split(img)
    B = (B - np.mean(B))
    G = (G - np.mean(G))
    R = (R - np.mean(R))
    img_new = cv2.merge([B, G, R])'''
    img_new = img / 255.0
    return img_new

def default_loader(filename,root1,root2,root3,root4,root5):
    pass
    return








class ImageFolder(data.Dataset):

    def __init__(self, trainlist):
        table = pd.read_table(trainlist, header=None, sep=',')
        table = table.sample(frac=1.0)
        self.trainlist = table.values
        self.len=self.trainlist.shape[0]
        self.GLEN=978
        #self.loader = default_loader

    def __getitem__(self, index):
        img1= np.zeros([1024],np.float32)
        img1[:self.GLEN]=self.trainlist[index][:self.GLEN]
        img1=img1.reshape([1,32,32])
        label1=np.array([self.trainlist[index][self.GLEN]],np.int64)
        img2 = np.zeros([1024], np.float32)
        img2[:self.GLEN] = self.trainlist[index + 1][:self.GLEN]
        img2=img2.reshape([1,32,32])
        label2=np.array([self.trainlist[index + 1][self.GLEN]],np.int64)
        img1 = torch.Tensor(img1)
        img2 = torch.Tensor(img2)
        label1 = torch.LongTensor(label1)
        label2 = torch.LongTensor(label2)
        '''if(self.trainlist[index][self.GLEN] != self.trainlist[index +1 ][self.GLEN]):
            label=np.array([1],np.int64)#not same class
        else:
            label=np.array([0],np.int64)#same class
        label = torch.LongTensor(label)'''


        return img1,img2,label1,label2

    def __len__(self):
        return 124 * 16 #self.len - 2