import numpy as np
import matplotlib.pyplot as plt
A = np.random.rand(30,10)
b = np.random.rand(30,1)
while np.linalg.matrix_rank(A)<10:
    A = np.random.rand(30,10)

x =  np.zeros((10,1))
X =[]
Y = []
for i in range(100):
    oldx = x
    x = x-np.dot(A.T,(np.dot(A,x)-b))/(np.linalg.norm(A)**2)
    diff = np.linalg.norm(oldx-x)
    print('Iteration ',i,'Norm of Difference between old and new x: ',diff)
    X.append(i)
    Y.append(diff)
plt.plot(X,Y)
plt.show()

