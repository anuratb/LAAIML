

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = [10,10]

fig = plt.figure()

r = 1
#u,v = np.mgrid[0:2 * np.pi:100j,0:np.pi:100j]
u = np.mgrid[0:2*np.pi:100j]
#print(u.shape)
#X = np.array([np.cos(u)*np.sin(v) ,np.sin(u)*np.sin(v),np.cos(v) ])
X = np.array([np.cos(u),np.sin(u)]).reshape((2,100))
#X = X.reshape((3,1,10000))

#ax = plt.axes(projection='3d')
#ax.scatter3D(X[0], X[1], X[2])
#print(X)
plt.plot(X[0],X[1])
plt.show()
X = X.reshape((2,1,100))
#x = X[0,0]
#y = X[1,0]

#axes.plot(x, y)
#plt.show()
A = np.array([[1,1],[1,0]]).reshape((2,2))#input matrix
#Print condition number
print(np.linalg.cond(A))

#print determinant
print('Determinant : ',np.linalg.det(A))
Y = np.array([np.dot(A,X[:,:,i]) for i in range(100)]).T.reshape(2,100)#do transformation



x = Y[0]
y = Y[1]
#z = Y[2]

print(x.shape)

#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z)
plt.plot(x,y)
plt.show()

