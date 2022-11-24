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


class KLR:

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

        m = len(self.y)
        theta = cp.Variable((m + 1, 1))
        C_parameter = cp.Parameter(nonneg=True)
        # CHANGED
        C_parameter.value = self.C
        lambd = cp.Parameter(nonneg=True)
        lambd.value = 1.0 / (2.0 * m * self.C)

        # eta = cp.multiply(self.y[:, None], (theta[-1] - (K @ (cp.multiply(theta[:m], self.y[:, None])))))
        eta = cp.multiply(-self.y, (K @ theta[:m] + theta[-1])[:, 0])

        loss = cp.sum(
            C_parameter * cp.logistic(eta)
        )
        objective_1 = cp.quad_form(cp.multiply(theta[:m], self.y[:, None]), psd_wrap(K))
        objective_2 = cp.quad_form(theta[:m], psd_wrap(K))

        # CHANGED
        # removed 0.5 * objective_2
        prob = cp.Problem(cp.Minimize(loss + 0.5*objective_2))

        try:
            prob.solve(solver='ECOS', abstol=1e-6)
        except SolverError:
            # TODO: use sklearn then
            prob.solve(solver='SCS')

        self.weights = theta.value[:-1].T
        self.bias = theta.value[-1]

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
class KNNC:

    def __init__(self, X, y):
        self.X = X
        self.kdt = KDTree(X, metric='euclidean')

        # self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X, k):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(k, len(self.X) - 1) + 1, return_distance=True)
        return dist[0, [[-1]]]
        #return dist[:, 1:].mean(axis=1)[:, None]

    # def predict(self, X):
    #    return self.model.predict(X)


class KNNM:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kdt = KDTree(X, metric='euclidean')
        # self.model = KNeighborsClassifier(n_neighbors=min(int(k), len(X) - 1))
        # self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X, k):
        i = self.kdt.query(X, k=min(k, len(self.X) - 1) + 1, return_distance=False)
        mp = self.X[i[:, 1:].flatten()].mean(axis=0)
        return np.linalg.norm(X - mp, axis=1)[:, None]

    def predict(self, X):
        return self.model.predict(X)

class KNNC_inspect:

    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.kdt = KDTree(X, metric='euclidean')

        # self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(self.k, len(self.X) - 1) + 1, return_distance=True)
        return dist[0, [[-1]]]

    def decision_function(self, X):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(self.k, len(self.X) - 1) + 1, return_distance=True)
        return dist[:, -1]
        #return dist[:, 1:].mean(axis=1)[:, None]

    # def predict(self, X):
    #    return self.model.predict(X)