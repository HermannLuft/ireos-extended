import mosek
import numpy as np
import pandas as pd
from cvxpy import SolverError
from numba import njit, jit, float64
from numba.experimental import jitclass
from optimparallel import minimize_parallel
# from rpy2 import robjects
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, KDTree
from cvxpy.atoms.affine.wraps import psd_wrap
from kernel import create_kernel
import cvxpy as cp
import numpy.typing as npt

"""
Module implementing a part of the separability algorithms
"""


class KLR:
    """
    Kernel Logistic Regression
    Works under the convex optimization framework for Python: cvxpy

    Application example:
    model = KLR(C=100, gamma=1.0)
    model.fit(X, y)
    model.predict_proba(test_sample)
    """

    def __init__(self, kernel='rbf', C=None, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.weights = None
        self.X = None
        self.y = None
        self.bias = 0
        self.kernel_function = lambda x: x

    def fit(self, X, y):
        """
        Estimating parameters beta and bias for data X and classes y
        """

        # adjust y for simple loss function
        self.y = np.copy(y)
        if np.logical_or(y == 0, y == 1).all():
            self.y[y == 0] = -1
        self.X = X

        # create Kernel matrix K and summation function over K(x_j, input)
        K, self.kernel_function = create_kernel(X, self.gamma, kernel=self.kernel)

        # creating parameters
        m = len(self.y)
        beta = cp.Variable((m + 1, 1))
        C_parameter = cp.Parameter(nonneg=True)
        C_parameter.value = self.C

        # classification factor: sum over ln(1 + exp(-y_i*f_i))
        eta = cp.multiply(-self.y, (K @ beta[:m] + beta[-1])[:, 0])
        loss = cp.sum(
            C_parameter * cp.logistic(eta)
        )

        # regularization factor: ||beta^T @ K @ beta||
        objective = cp.quad_form(beta[:m], psd_wrap(K))

        # as we minimize the soft margin problem
        constraints = []

        prob = cp.Problem(cp.Minimize(loss + 0.5 * objective), constraints)

        # Problem solving with FALLBACK MOSEK or when no licence obtained: ECOS -> SCS
        try:
            prob.solve(solver='MOSEK', mosek_params={mosek.iparam.bi_max_iterations: 10_000,
                                                     mosek.iparam.intpnt_max_iterations: 10_000,
                                                     mosek.iparam.sim_max_iterations: 10_000})
            # prob.solve(solver='ECOS', abstol=1e-6)
        except SolverError:
            prob.solve(solver='SCS', verbose=False)

        self.weights = beta.value[:-1].T
        self.bias = beta.value[-1]

    def decision_function(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (self.weights @ self.kernel_function(self.X, x) + self.bias)[0]

    def predict_proba(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # TODO: test with multiple entries in x
        pos_prob = 1.0 / (1.0 + np.exp(-self.decision_function(x)))
        neg_prob = 1.0 - pos_prob
        return np.hstack((neg_prob[:, None], pos_prob[:, None]))

    def predict(self, x):
        predictions = self.predict_proba(x)[:, 0]
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        return predictions

    def get_gamma(self):
        return self.gamma


class KLR_alt:
    """
    Kernel Logistic Regression based on scipy optimization
    """

    @staticmethod
    # @DeprecationWarning
    def _optimizing_function_R(beta, y, K, C):
        """
        equal implementation to calibrateBinary in R
        """
        weights = beta[None, :-1]
        bias = beta[-1]
        regularization_factor = 0.5 * (weights @ K @ weights.T)[0, 0]
        loss_function = -y * (K @ weights.T + bias)[:, 0]  # - y*f
        classification_factor = (1.0 / len(y)) * np.sum(C * np.log(1.0 + np.exp(loss_function)))  # loss(-y*f)
        return regularization_factor + classification_factor

    @staticmethod
    # @DeprecationWarning
    @njit(fastmath=True)
    def _optimizing_function_Paper_nopython(beta, y, K, C):
        """
        optimized loss function
        """
        weights = np.reshape(beta[:-1], (1, len(y)))
        bias = beta[-1]
        regularization_factor = 0.5 * (weights @ K @ weights.T)[0, 0]
        loss_function = -y * (K @ weights.T + bias)[:, 0]  # - y*f
        classification_factor = np.sum(C * np.log(1.0 + np.exp(loss_function)))  # loss(-y*f)
        return classification_factor + regularization_factor

    @staticmethod
    @njit(fastmath=True)
    def _optimizing_function_data_independent_nopython(beta, y, K, C):
        '''
        optimized loss function adjusted to datasize
        '''
        weights = np.reshape(beta[:-1], (1, len(y)))
        bias = beta[-1]
        regularization_factor = (1.0 / len(y)) * 0.5 * (weights @ K @ weights.T)[0, 0]
        loss_function = -y * (K @ weights.T + bias)[:, 0]
        classification_factor = np.sum(C * np.log(1.0 + np.exp(loss_function)))
        return regularization_factor + classification_factor

    def __init__(self, kernel='rbf', C=None, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.weights = None
        self.X = None
        self.y = None
        self.bias = 0
        self.kernel_function = lambda x: x

    def fit(self, X, y):
        self.y = np.copy(y)
        if np.logical_or(y == 0, y == 1).all():
            self.y[y == 0] = -1
        self.X = X

        K, self.kernel_function = create_kernel(X, self.gamma, kernel=self.kernel)
        # beta0 = np.random.normal(0, 1, len(y) + 1)
        beta0 = np.ones(len(y) + 1)

        res = minimize(self._optimizing_function_Paper_nopython, method="SLSQP",
                       x0=beta0, options={'disp': False}, args=(self.y, K, self.C),
                       )

        self.weights = res.x[None, :-1]
        self.bias = res.x[-1]

        return res.x

    def decision_function(self, x):
        return (self.weights @ self.kernel_function(self.X, x) + self.bias)[0]

    def predict_proba(self, x):
        # TODO: test with multiple entries in x
        pos_prob = 1.0 / (1.0 + np.exp(-self.decision_function(x)))
        neg_prob = 1.0 - pos_prob
        return np.hstack((neg_prob[:, None], pos_prob[:, None]))

    def predict(self, x):
        predictions = self.predict_proba(x)[:, 0]
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        return predictions


# Information: kNN classifier aus sklearn interessant
class KNNM:
    """
    Implementing kNN maximum margin approximator based on means
    """

    def __init__(self, k):
        self.kdt = None
        self.X = None
        self.k = k

    def get_k(self):
        return self.k
        # self.model.fit(X=X[y == 0], y=y[y == 0])

    def fit(self, X, *args):
        self.X = X
        self.kdt = KDTree(X, metric='euclidean')

    def decision_function(self, X):
        i = self.kdt.query(X, k=min(self.k, len(self.X) - 1) + 1, return_distance=False)
        mp = self.X[i[:, 1:].flatten()].mean(axis=0)
        return np.linalg.norm(X - mp, axis=1)[:, None]


class KNNC:
    """
    Implementing kNN maximum margin approximator based on distances
    """

    def __init__(self, k):
        self.kdt = None
        self.X = None
        self.k = k

    def get_k(self):
        return self.k

    def fit(self, X, *args):
        self.X = X
        self.kdt = KDTree(X, metric='euclidean')

    def predict_proba(self, X):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(self.k, len(self.X) - 1) + 1, return_distance=True)
        return dist[0, [[-1]]]

    def decision_function(self, X):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(self.k, len(self.X) - 1) + 1, return_distance=True)
        return dist[:, -1]
        # return dist[:, 1:].mean(axis=1)[:, None]

    # def predict(self, X):
    #    return self.model.predict(X)


class KNNC_w:
    """
    Implementing kNN maximum margin approximator based on distances to inlier neighbors
    """


    def __init__(self, k):
        self.kdt = None
        self.X = None
        self.k = k

    def get_k(self):
        return self.k

    def fit(self, X, w, *args):
        self.X = X[w < 0.5]
        self.kdt = KDTree(X[w < 0.5], metric='euclidean')

    def predict_proba(self, sample):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(sample, k=min(self.k, len(self.X) - 1), return_distance=True)
        return dist[0, [[-1]]]

    def decision_function(self, sample):
        # dist, _ = self.model.kneighbors(X)
        neighbor_addon = 1 if sample in self.X else 0
        dist, _ = self.kdt.query(sample, k=min(self.k, len(self.X) - 1) + neighbor_addon, return_distance=True)
        return dist[:, -1]
