import numpy as np
from keras.datasets import mnist
(train_X,train_y),(val_X,val_y) = mnist.load_data()

'''
columns_taken = []
ind = 0
X = []
 #remove the 0 columns to make the matrix with L.I. conlumns
for x in train_X.T:
  if np.sum(x)>0:
    X.append(x)
    columns_taken.append(ind)
  ind+=1
X.append([1]*X[0].shape[0])#add bias
train_X = np.array(X).T
'''

#sample test cases
#choices = np.random.choice(val_X.shape[0], 1000, replace=False)
#val_X,val_y = val_X[choices, :],val_y[choices]
#remove those columns from X which were removed earlier
#val_X = val_X.T[columns_taken].T  
#also add bias column
#bias = np.ones((1000,1))
#val_X = np.append(val_X,bias,axis = 1)
#find confusion matrix