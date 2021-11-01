import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('test.csv', header = None, names = ['garbage'])#, skiprows = 1, names = ['a','b','c','d','e','f'])
train_test = train['garbage'].map(lambda x:x.split(' '))
#  pd.Cov 
# train.columns = ['a','c','c','c','c','c','c','c','c','c','c','c','c','c','a','c','c','c','c','c','c','c','c','c','c','c','c','c']
# train = pd.DataFrame(data = train, columns = ['a','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c'])
print(train_test.head())