import pandas as pd
import numpy as np

train = pd.read_csv("Train.csv",delimiter = ',')
train_item_list = train['Item_Fat_Content'].unique()
print(train_item_list)
train.replace('reg','Regular').replace(['low fat','LF'],'Low Fat')
data = train['Item_Fat_Content'].head()
print(data)



# train.shape
# pd.set_option('display.max_columns', None)
# # print(train.head(2))
# imp = SimpleImputer(strategy='mean')
# train_test = imp.fit(train)
# print(train_test)













