import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import linear_kernel

"""
This model implements kernel leverage with additional dimensions or kernel matrix
"""


def basis_poly_function_2d(X):
    # leverage in polynomial dimensions of input
    X = np.hstack((X, (X[:, 0] * X[:, 0])[:, None]))
    X = np.hstack((X, (X[:, 0] * X[:, 1])[:, None]))
    X = np.hstack((X, (X[:, 1] * X[:, 1])[:, None]))
    return X


def create_kernel(data, gamma=1, kernel='rbf'):
    # Returns Kernel matrix K with kernel function k(x_{}, x_{})
    length_scale = np.sqrt(1.0 / (2.0 * gamma))

    kernel_function = linear_kernel
    if kernel == 'rbf':
        kernel_function = RBF(length_scale)
    return kernel_function(data), kernel_function