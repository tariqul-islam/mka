import sys
sys.path.append("../codes")

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def get_gaussian_exp_data(N,D,s):
    X = np.random.randn(N,D)
    Y = np.random.randn(N,D)
    Z = X + s * np.random.randn(N,D)
    
    return X, Y, Z

#Let's visualize in 2D
X,Y,Z = get_gaussian_exp_data(5000,2,0.5)

X_0 = np.argsort(X[:,0])


y = np.zeros(len(X_0))
print(y)
for i in range(len(X_0)):
    y[X_0[i]] = i
    
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, s=2)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.axis('off')
plt.title('Gaussian Spot', fontsize=35)

plt.figure()
plt.scatter(Y[:,0], Y[:,1], c=y, s=2)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('Lost-correspondence', fontsize=35)
plt.axis('off')

plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=y, s=2)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('Perturbed', fontsize=35)
plt.axis('off')


from mka import MKA, kCKA, CKA_RBF

X,Y,Z = get_gaussian_exp_data(5000,200,0.5) #Gaussian of 200 dimensions

print('Lost Correspondence:')
print('MKA: ', MKA(X,Y, n_neighbors = 200))
print('kCKA: ', kCKA(X,Y, n_neighbors = 200))
print('CKA: ', CKA_RBF(X,Y))

print('Purturbed:')
print('MKA: ', MKA(X,Z, n_neighbors = 200))
print('kCKA: ', kCKA(X,Z, n_neighbors = 200))
print('CKA: ', CKA_RBF(X,Z))

