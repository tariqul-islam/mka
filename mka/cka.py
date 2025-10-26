    #https://goodresearch.dev/cka.html

import numpy as np

import numba
from numba import prange

import random

from .helper import euclidean_distances_numba, frobenius_norm_sq

@numba.jit(nopython=True)
def get_median_dists(x):
    N = len(x)
    sorted_x = np.sort(x.reshape(-1))[N:]
    N = len(sorted_x)
    median = 0.5 * (sorted_x[N//2-1]+sorted_x[N//2])
    if median < 10**-12:
        median = 0.1
    
    return median
    
    

@numba.jit(nopython=True)
def cka_rbf_kernel_matrix(X, si = None):
    dists = euclidean_distances_numba(X)
    #print(dists)
    sigma = np.median(dists) #np.median(dists) #get_median_dists(dists)
    #print(sigma)
    
    if sigma < 10**-12:
        sigma = 1.0
    
    if si is not None:
        #print('in block')
        sigma = si**2*sigma
    
    #print(sigma)
    kernel = np.exp(-dists/(2*sigma))
    #print(kernel)
    return kernel
    

@numba.jit(nopython=True)    
def cka_rbf(X,Y,si=None):
    N = len(X)
    
    X = X #- np.mean(X,axis=0)
    Y = Y #- np.mean(Y,axis=0)

    K = cka_rbf_kernel_matrix(X,si)
    L = cka_rbf_kernel_matrix(Y,si)
    
    #print(K,L)
    
    H = np.eye(N)-1/N * np.ones((N,N))
    
    f = lambda P,Q: np.trace(P.dot(H).dot(Q).dot(H))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    score = KHLH / np.sqrt(KHKH*LHLH)
    
    return score, K, L

@numba.jit(nopython=True)
def tcka_rbf_kernel_matrix(X, n_neighbors=15):
    N = len(X)
    dists = euclidean_distances_numba(X)
    
    sort_idx = np.zeros((N,n_neighbors), dtype=numba.int64)
    dists_2 = np.zeros((N, n_neighbors))
    
    for i in range(N):
        sort_idx[i,:] = np.argsort(dists[i])[1:n_neighbors+1]
        dists_2[i,:] = dists[i,sort_idx[i,:]]
    
    #sort_idx = np.argsort(dists, axis=1)
    #sort_idx = sort_idx.astype(np.int32)
    #sort_idx = sort_idx[:,1:n_neighbors+1]
    #for i in range(N):
    #    dists_2[i,:] = dists[i,sort_idx[i,:]]

    sigma = np.median(dists_2) 
    if sigma < 10**-12:
        sigma = 1.0   
    k = np.exp(-dists_2/(2*sigma))

    kernel = np.zeros((N,N))
    
    for i in range(len(X)):
        kernel[i, sort_idx[i]] = k[i]
    
    return kernel   
    
     
    
#@numba.jit(nopython=True)    
def tcka_rbf(X,Y, n_neighbors=15, diag=True):
    N = len(X)
    
    X = X #- np.mean(X,axis=0)
    Y = Y #- np.mean(Y,axis=0)

    K = tcka_rbf_kernel_matrix(X, n_neighbors=n_neighbors)
    L = tcka_rbf_kernel_matrix(Y, n_neighbors=n_neighbors)
    
    if diag:
        for i in range(N):
            K[i,i] = 1
            L[i,i] = 1
    
    H = np.eye(N)-1/N * np.ones((N,N))
    
    f = lambda P,Q: np.trace(P.dot(H).dot(Q).dot(H))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    score = KHLH / np.sqrt(KHKH*LHLH)
    
    return score, K, L
    
def kCKA(X,Y, n_neighbors=15, diag=True, get_matrices=False):
    score, K, L = tcka_rbf(X,Y, n_neighbors=n_neighbors, diag=diag)
    
    if get_matrices:
        return score, K, L
    else:
        return score
        
def cka_rbf_lite(X,Y,si=None, get_matrices=False):
    score, K, L = cka_rbf(X,Y,si=si)
    if get_matrices:
        return score, K, L
    else:
        return score
            
    
