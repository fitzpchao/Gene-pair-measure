import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import random
from osgeo import gdalnumeric
import os

class ImageFolder(data.Dataset):

    def __init__(self, trainlist):
        self.table = pd.read_table(trainlist, header=None, sep=',')
        self.trainlist = self.table.values
        self.len=self.trainlist.shape[0]
        print(trainlist,self.trainlist.shape[1])
        self.GLEN=978
        #self.loader = default_loader

    def __getitem__(self, index):
        self.table = self.table.sample(frac=1.0)
        self.trainlist= self.table.values
        img1= np.zeros([1024],np.float32)
        img1[:self.GLEN]=self.trainlist[index][:self.GLEN]
        img1=img1.reshape([1,32,32])
        #label1=np.array([self.trainlist[index][self.GLEN]],np.int64)
        img2 = np.zeros([1024], np.float32)
        img2[:self.GLEN] = self.trainlist[index + 1][:self.GLEN]
        img2=img2.reshape([1,32,32])
        #label2=np.array([self.trainlist[index + 1][self.GLEN]],np.int64)
        img1 = torch.Tensor(img1)
        img2 = torch.Tensor(img2)
        #label1 = torch.LongTensor(label1)
        #label2 = torch.LongTensor(label2)
        #print(self.trainlist[index][self.GLEN])
        #print(self.trainlist[index + 1][self.GLEN])
        if(self.trainlist[index][self.GLEN] != self.trainlist[index +1 ][self.GLEN]):
            label=np.array([1],np.int64)#not same class
        else:
            label=np.array([0],np.int64)#same class
        #print(label)
        label = torch.LongTensor(label)


        return img1,img2,label

    def __len__(self):
        return int((self.len - 1) / 1 ) * 1 #self.len - 2

class ImageFolder_val(data.Dataset):

    def __init__(self, trainlist):
        self.table = pd.read_table(trainlist, header=None, sep=',')
        self.trainlist = self.table.values
        self.len=self.trainlist.shape[0]
        self.GLEN=978
        #self.loader = default_loader

    def __getitem__(self, index):
        self.table = self.table.sample(frac=1.0)
        self.trainlist= self.table.values
        img1= np.zeros([1024],np.float32)
        img1[:self.GLEN]=self.trainlist[index][:self.GLEN]
        img1=img1.reshape([1,32,32])
        #label1=np.array([self.trainlist[index][self.GLEN]],np.int64)
        img2 = np.zeros([1024], np.float32)
        img2[:self.GLEN] = self.trainlist[index + 1][:self.GLEN]
        img2=img2.reshape([1,32,32])
        #label2=np.array([self.trainlist[index + 1][self.GLEN]],np.int64)
        img1 = torch.Tensor(img1)
        img2 = torch.Tensor(img2)
        #label1 = torch.LongTensor(label1)
        #label2 = torch.LongTensor(label2)
        #print(self.trainlist[index][self.GLEN])
        #print(self.trainlist[index + 1 ][self.GLEN])
        if(self.trainlist[index][self.GLEN] != self.trainlist[index +1 ][self.GLEN]):
            label=np.array([1],np.int64)#not same class
        else:
            label=np.array([0],np.int64)#same class
        #print(label)
        label = torch.LongTensor(label)


        return img1,img2,label

    def __len__(self):
        return int((self.len - 1) / 16 ) * 16 #self.len - 2