import numpy as np

import numba
from numba import prange

import random

from .umap_pre import get_prob_matrix

def umap_cka(X,Y, n_neighbors=15, diag=False, symmetry=True):
    N = len(X)
    
    X = X 
    Y = Y 

    K,_ = get_prob_matrix(X, n_neighbors=n_neighbors, symmetry=symmetry)
    L,_ = get_prob_matrix(Y, n_neighbors=n_neighbors, symmetry=symmetry)
    
    if diag:
        for i in range(N):
            K[i,i] = 1
            L[i,i] = 1
        
    
    H = np.eye(N)-1/N * np.ones((N,N))
    
    f = lambda P: P.dot(H)
    
    KH = f(K)
    LH = f(L)
    
    KHLH = np.trace(KH.dot(LH))
    KHKH = np.trace(KH.dot(KH))
    LHLH = np.trace(LH.dot(LH))
    
    score = KHLH / np.sqrt(KHKH*LHLH)
    
    return score, K, L
    
    
def umap_cka_nonsym(X,Y, n_neighbors=15, diag=False):
    N = len(X)
    
    X = X 
    Y = Y 

    K,_ = get_prob_matrix(X, n_neighbors=n_neighbors, symmetry=False)
    L,_ = get_prob_matrix(Y, n_neighbors=n_neighbors, symmetry=False)
    
    if diag:
        for i in range(N):
            K[i,i] = 1
            L[i,i] = 1
        
    
    H = np.eye(N)-1/N * np.ones((N,N))
    
    def f (P,Q):
        J = P.dot(H)
        L = Q.dot(H)
        return np.trace(J.dot(L.T))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    print('in function: ', KHLH, KHKH, LHLH)
    
    score = KHLH / np.sqrt(KHKH*LHLH)
    
    return score, K, L
    
    

    
    
def umap_cka_nonsym_smiple(X,Y, n_neighbors=15, diag=True):
    N = len(X)
    
    X = X 
    Y = Y 

    K,_ = get_prob_matrix(X, n_neighbors=n_neighbors, symmetry=False)
    L,_ = get_prob_matrix(Y, n_neighbors=n_neighbors, symmetry=False)
    
    D2 = (np.log2(n_neighbors))**2
    
    if diag:
        D2 = (1+np.log2(n_neighbors))**2
        for i in range(N):
            K[i,i] = 1
            L[i,i] = 1
    
    f = lambda P,Q: np.sum(P*Q) #np.trace(P.dot(Q.T))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    #print('in function: ', KHLH-D2, KHKH-D2, LHLH-D2)
    
    score = (KHLH-D2) / np.sqrt((KHKH-D2)*(LHLH-D2))
    
    return score, K, L
    
    
def MKA(X,Y, n_neighbors=15, diag=True, get_matrices=False):
    score, K, L = umap_cka_nonsym_smiple(X, Y, n_neighbors=n_neighbors, 
                                         diag=diag)
    
    if get_matrices:
        return score, K, L
    else:
        return score
    
def umap_align(X,Y, n_neighbors=15, diag=False):
    N = len(X)
    
    X = X 
    Y = Y 

    K,_ = get_prob_matrix(X, n_neighbors=n_neighbors, symmetry=False)
    L,_ = get_prob_matrix(Y, n_neighbors=n_neighbors, symmetry=False)
    
    if diag:
        for i in range(N):
            K[i,i] = 1
            L[i,i] = 1
    
    f = lambda P,Q: np.sum(P*Q) #np.trace(P.dot(Q.T))
    
    KHLH = f(K,L)
    KHKH = f(K,K)
    LHLH = f(L,L)
    
    #print('in function: ', KHLH-D2, KHKH-D2, LHLH-D2)
    
    score = (KHLH) / np.sqrt((KHKH)*(LHLH))
    
    return score, K, L
