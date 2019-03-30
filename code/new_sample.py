import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
LEN=1500
TEST_SIZE=0.5
F_LEN=12328
LEN_TRAIN =int( LEN * (1- TEST_SIZE))
LEN_TEST = int(LEN * TEST_SIZE)
def readTable(name):
    table = pd.read_table(name,sep=',',index_col=0)
    #table = table.sample(frac=1.0)
    table = table.values
    return table
def saveTable(data,name):
    table=pd.DataFrame(data)
    table.to_csv(name,header=None,index=None)

listname=os.listdir('data2')
print(listname)
tables=[]
trainTable=np.zeros([LEN_TRAIN,12329],np.float32)
validationTable=np.zeros([LEN_TEST,12329],np.float32)
for i in range(15):
    csv='data2/'+ listname[i]
    tables.append(readTable(csv))

table = np.concatenate(tables,axis=0)
print(table.shape)
inds = np.arange(LEN)
print(inds)
print(inds.shape)
inds_train, inds_test = train_test_split(inds, test_size=TEST_SIZE)
np.save('inds_train_9.npy',inds_train)
np.save('inds_test_9.npy',inds_test)


for i in range(inds_train.shape[0]):
        trainTable[i][:F_LEN]=table[inds_train[i]]
        trainTable[i][F_LEN]= int(inds_train[i]/100)

for i in range(inds_test.shape[0]):
        validationTable[i][:F_LEN]=table[inds_test[i]]
        validationTable[i][F_LEN] = int(inds_test[i]/100)


saveTable(trainTable,'trainsample_9.csv')
saveTable(validationTable,'validationsample_9.csv')