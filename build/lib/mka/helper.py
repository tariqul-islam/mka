import numpy as np

import numba
from numba import prange

import random

import scipy.sparse

import gc

@numba.jit(nopython=True, parallel=True)
def euclidean_distances_numba(X, squared = True):
    n = X.shape[0]
    xcorr = np.zeros((n,n),dtype=X.dtype)
    for i in prange(n):
        for j in range(i,n):
            dist = np.sum( np.square(X[i,:] - X[j,:]) )
            if not squared:
                dist = np.sqrt(dist)
            xcorr[i,j] = dist
            xcorr[j,i] = dist
    
    return xcorr
    
@numba.jit(nopython=True)
def radial_basis(d,sigma):
    return np.exp(-d/(2*sigma)) #both d and sigma are squared
    
@numba.jit(nopython=True)
def frobenius_norm_sq(X):
    return np.sum(np.square(X))
