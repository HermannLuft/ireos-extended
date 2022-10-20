import numpy as np
from numba import njit
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import linear_kernel
import tensorflow_probability as tfp


def basis_poly_function_2d(X):
    X = np.hstack((X, (X[:, 0] * X[:, 0])[:, None]))
    X = np.hstack((X, (X[:, 0] * X[:, 1])[:, None]))
    X = np.hstack((X, (X[:, 1] * X[:, 1])[:, None]))
    return X


def create_kernel(data, gamma=1, kernel='rbf'):
    '''
    Returns K, k(_, _)
    '''
    length_scale = np.sqrt(1.0 / (2.0 * gamma))
    # linear kernel
    kernel_function = linear_kernel
    if kernel == 'rbf':
        kernel_function = RBF(length_scale)
    return kernel_function(data), kernel_function