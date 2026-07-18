    #https://goodresearch.dev/cka.html

import numpy as np

import numba
from numba import prange

import random

from .helper import euclidean_distances_numba, frobenius_norm_sq

from sklearn.metrics.pairwise import euclidean_distances

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
def linear_kernel_matrix(X):
    return X.dot(X.T)

    

#@numba.jit(nopython=True)
def cka_rbf_kernel_matrix(X, si = None):
    dists = euclidean_distances(
        X,
        X,
        squared=True,
    )
    
    sigma = np.median(dists) #np.median(dists) #get_median_dists(dists)
    
    if sigma < 10**-12:
        sigma = 1.0
    
    if si is not None:
        #print('in block')
        sigma = si**2*sigma
    
    kernel = np.exp(-dists/(2*sigma))

    return kernel
    

@numba.jit(nopython=True)    
def cka_rbf(X,Y,si=None, diag=True):
    N = len(X)
    
    X = X #- np.mean(X,axis=0)
    Y = Y #- np.mean(Y,axis=0)

    K = cka_rbf_kernel_matrix(X,si)
    L = cka_rbf_kernel_matrix(Y,si)
    
    if not diag:
        for i in range(N):
            K[i,i] = 0
            L[i,i] = 0
    
    #print(K,L)
    
    H = np.eye(N)-1/N * np.ones((N,N))
    
    f = lambda P,Q: np.trace(P.dot(H).dot(Q).dot(H))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    score = KHLH / np.sqrt(KHKH*LHLH)
    
    return score, K, L

#@numba.jit(nopython=True)
def tcka_rbf_kernel_matrix(X, n_neighbors=15):
    N = len(X)
    dists = euclidean_distances_numba(X)
    
    #sort_idx = np.zeros((N,n_neighbors)
    #dists_2 = np.zeros((N, n_neighbors))
    
    sort_idx = np.argsort(dists, axis=1)

    # Drop the first entry, assumed to be the point itself.
    sort_idx = sort_idx[:, 1:n_neighbors + 1]

    # Gather the corresponding distances.
    dists_2 = np.take_along_axis(
        dists,
        sort_idx,
        axis=1,
    )
    
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
        
def cka_rbf_lite(X,Y,si=None, diag=True, get_matrices=False):
    score, K, L = cka_rbf(X,Y,si=si, diag=diag)
    if get_matrices:
        return score, K, L
    else:
        return score

@numba.jit(nopython=True)
def topk_mask_from_kernel(K, n_neighbors=15, exclude_diag=True):
    N = K.shape[0]
    mask = np.zeros((N, N))

    for i in range(N):
        row = K[i].copy()

        if exclude_diag:
            row[i] = -np.inf

        sort_idx = np.argsort(row)[::-1]
        for j in range(n_neighbors):
            mask[i, sort_idx[j]] = 1.0

    return mask


@numba.jit(nopython=True)
def hsic_biased_masked(K, L):
    N = K.shape[0]
    H = np.eye(N) - 1.0 / N * np.ones((N, N))
    return np.trace(K.dot(H).dot(L).dot(H))


@numba.jit(nopython=True)
def hsic_unbiased_masked(K, L):
    N = K.shape[0]

    if N <= 3:
        return np.nan

    K_tilde = K.copy()
    L_tilde = L.copy()

    for i in range(N):
        K_tilde[i, i] = 0.0
        L_tilde[i, i] = 0.0

    term_1 = np.sum(K_tilde * L_tilde.T)
    term_2 = np.sum(K_tilde) * np.sum(L_tilde) / ((N - 1) * (N - 2))
    term_3 = 2.0 * np.sum(K_tilde.dot(L_tilde)) / (N - 2)

    return (term_1 + term_2 - term_3) / (N * (N - 3))


@numba.jit(nopython=True)
def cknna_similarity(K, L, n_neighbors=15, unbiased=False):
    mask_K = topk_mask_from_kernel(K, n_neighbors=n_neighbors, exclude_diag=True)
    mask_L = topk_mask_from_kernel(L, n_neighbors=n_neighbors, exclude_diag=True)

    # CKNNA: only pairs that are nearest neighbors in both kernels contribute.
    mask = mask_K * mask_L

    K_masked = mask * K
    L_masked = mask * L

    if unbiased:
        return hsic_unbiased_masked(K_masked, L_masked)
    else:
        return hsic_biased_masked(K_masked, L_masked)


def cknna(X, Y, n_neighbors=15, kernel="rbf", si=None, diag=True):
    N = len(X)

    if n_neighbors < 2:
        raise ValueError("CKNNA requires n_neighbors >= 2.")

    if n_neighbors >= N:
        n_neighbors = N - 1

    if kernel == "rbf":
        # Reuses your existing CKA RBF kernel construction.
        K = cka_rbf_kernel_matrix(X, si=si)
        L = cka_rbf_kernel_matrix(Y, si=si)

    elif kernel == "linear":
        # This is closer to the PRH implementation, which uses inner products.
        K = linear_kernel_matrix(X)
        L = linear_kernel_matrix(Y)

    else:
        raise ValueError("kernel must be either 'rbf' or 'linear'.")

    if not diag:
        for i in range(N):
            K[i, i] = 0.0
            L[i, i] = 0.0

    mask_K = topk_mask_from_kernel(K, n_neighbors=n_neighbors, exclude_diag=True)
    mask_L = topk_mask_from_kernel(L, n_neighbors=n_neighbors, exclude_diag=True)
    mask = mask_K * mask_L

    K_masked = mask * K
    L_masked = mask * L
    K = K * mask_K
    L = L * mask_L

    H = np.eye(N) - 1.0 / N * np.ones((N, N))

    KHLH = np.trace(K_masked.dot(H).dot(L_masked).dot(H))
    KHKH = np.trace(K.dot(H).dot(K).dot(H))
    LHLH = np.trace(L.dot(H).dot(L).dot(H))

    score = KHLH / np.sqrt(KHKH * LHLH)

    return score, K, L


def CKNNA(
    X,
    Y,
    n_neighbors=15,
    kernel="rbf",
    si=None,
    diag=True,
    get_matrices=False,
):
    score, K, L = cknna(
        X,
        Y,
        n_neighbors=n_neighbors,
        kernel=kernel,
        si=si,
        diag=diag,
    )

    if get_matrices:
        return score, K, L
    else:
        return score
