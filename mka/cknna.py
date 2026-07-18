import numpy as np
import numba

from .cka import cka_rbf_kernel_matrix, linear_kernel_matrix


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
def row_center_kernel(K):
    N = K.shape[0]
    Kc = np.zeros((N, N))

    for i in range(N):
        row_mean = 0.0

        for j in range(N):
            row_mean += K[i, j]

        row_mean /= N

        for j in range(N):
            Kc[i, j] = K[i, j] - row_mean

    return Kc


@numba.jit(nopython=True)
def masked_frobenius_inner(Kc, Lc, mask):
    return np.sum(mask * Kc * Lc)


@numba.jit(nopython=True)
def cknna_from_kernels(K, L, n_neighbors=15):
    mask_K = topk_mask_from_kernel(K, n_neighbors=n_neighbors, exclude_diag=True)
    mask_L = topk_mask_from_kernel(L, n_neighbors=n_neighbors, exclude_diag=True)

    mutual_mask = mask_K * mask_L

    Kc = row_center_kernel(K)
    Lc = row_center_kernel(L)

    KHLH = masked_frobenius_inner(Kc, Lc, mutual_mask)
    KHKH = masked_frobenius_inner(Kc, Kc, mask_K)
    LHLH = masked_frobenius_inner(Lc, Lc, mask_L)

    denom = np.sqrt(KHKH * LHLH)

    if denom < 1e-12:
        return 0.0

    return KHLH / denom


def cknna(X, Y, n_neighbors=15, kernel="linear", si=None, diag=True):
    N = len(X)

    if n_neighbors < 1:
        raise ValueError("CKNNA requires n_neighbors >= 1.")

    if n_neighbors >= N:
        n_neighbors = N - 1

    if kernel == "linear":
        K = linear_kernel_matrix(X)
        L = linear_kernel_matrix(Y)

    elif kernel == "rbf":
        K = cka_rbf_kernel_matrix(X, si)
        L = cka_rbf_kernel_matrix(Y, si)

    else:
        raise ValueError("kernel must be either 'linear' or 'rbf'.")

    if not diag:
        for i in range(N):
            K[i, i] = 0.0
            L[i, i] = 0.0

    score = cknna_from_kernels(K, L, n_neighbors=n_neighbors)

    return score, K, L


def CKNNA(
    X,
    Y,
    n_neighbors=15,
    kernel="linear",
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
