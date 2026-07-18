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
    
def swiss_roll_1d(N=1000):
    r = np.linspace(0,1,N).reshape(1,N)
    t = 3*np.pi/2 * (1 + 2*r)
    return np.transpose(np.concatenate((t*np.cos(t),t*np.sin(t))))

def s_curve_1d(r_d=0.5, N = 1000):    
    r = np.linspace(0,1,N).reshape(1,N) #np.random.rand(1,N)
    t3 = 3 * np.pi * (r - r_d)
    return np.transpose(np.concatenate((np.sin(t3),np.sign(t3) * (np.cos(t3) - 1))))
    
def create_ring(N_points, R, noise=0):
    #Copied from RTD package
    arr = np.zeros((N_points, 2))

    for i in range(N_points):
        theta = 2 * np.pi * i / N_points
        d = (np.array([np.cos(theta), np.sin(theta)])
                 + np.random.multivariate_normal(np.zeros(2), noise * np.eye(2))) * R

        arr[i] = d

    return arr
    
def create_set_of_rings(N_large, N_total, noise = 0):
    #Copied from RTD package
    A = []

    for i in range(N_total):
        np.random.seed(i)
        if i < N_large:
            R = 1
        else:
            R = (i - N_large + 1) / (N_total - N_large + 1)
            idx = (i - N_large)
            
            if idx % 2 == 0:
                R = 1 + 0.25 * (2 + idx) / 2 
            else:
                R = 1 - 0.25 * (2+ idx - 1) / 2
            
        A.append(create_ring(100, R=R, noise=noise))
        
    A_mod = np.concatenate(A)
    
    return A_mod
    
    
def get_gaussian_exp_data(N,D,s):
    X = np.random.randn(N,D)
    X_lost_correspondance = np.random.randn(N,D)
    X_perturbed = X + s * np.random.randn(N,D)
    
    return X, X_lost_correspondance, X_perturbed
