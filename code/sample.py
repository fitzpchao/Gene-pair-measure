import pandas as pd

table=pd.read_csv('validationsample.csv',header=None,index_col=None,sep=',')
table=table.sample(frac=1.0)
table.to_csv('validation_shuffer.csv',header=None,index=None)