import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy.linalg as linalg

train = pd.read_csv('bank.csv',delimiter = ';',header = 'infer')
train_data = train.drop(['job','marital','education','default','housing','loan','contact','month','poutcome','y'],axis=1)
L2no = Normalizer()
L2_train = L2no.fit_transform(train_data)
# print(L2_train)

# variance after normal
L2_train.var(axis=0)

#normal + low variance + threshold
vt = VarianceThreshold(threshold=1e-04)
L2_train_vt = vt.fit_transform(L2_train)
print(L2_train_vt.var(axis=0))

#use PCA directly
pca = PCA(n_components=5)
L2_train_pca = pca.fit_transform(L2_train)
print(L2_train_pca.var(axis=0))


# step by step of PCA
cov_mat = np.cov(L2_train.T)
eig_vals , eig_vecs = linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), eig_pairs[1][1].reshape(7,1)
                     , eig_pairs[2][1].reshape(7,1), eig_pairs[3][1].reshape(7,1)
                        , eig_pairs[4][1].reshape(7,1)))

# Y = L2_train.dot(matrix_w)
# Y.shape
# Y.var(axis=0)


# #direct     versus    stpe bt step
# print('low variance =  ',L2_train_vt.var(axis=0))
# print('directly =      ',L2_train_pca.var(axis=0))
# print('step by step =  ',Y.var(axis=0))
