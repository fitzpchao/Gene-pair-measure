import pandas as pd
import numpy as np

def readTable(name):
    table = pd.read_table(name,sep=',',index_col=0)
    table = table.sample(frac=1.0)
    table = table.values
    return table
def saveTable(data,name):
    table=pd.DataFrame(data)
    table.to_csv(name,header=None,index=None)

csv1='untrt_MCF7_base_2000x978.csv'
csv2='untrt_PC3_base_2000x978.csv'
trainTable=np.zeros([2000,979],np.float32)
validationTable=np.zeros([2000,979],np.float32)
table1= readTable(csv1)
table2= readTable(csv2)
for i in range(len(table1)):
    if(i<1000):
        trainTable[i][:978]=table1[i]
        trainTable[i][978]=0
    else:
        validationTable[i-1000][:978] = table1[i]
        validationTable[i-1000][978]=0

for i in range(len(table2)):
    if(i<1000):
        trainTable[i + 1000][:978]=table2[i]
        trainTable[i + 1000][978]=1
    else:
        validationTable[i ][:978] = table2[i]
        validationTable[i ][978]=1

saveTable(trainTable,'trainsample.csv')
saveTable(validationTable,'validationsample.csv')