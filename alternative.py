from abc import ABC
from abc import ABC

import numpy as np
from PyNomaly import loop
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numba import njit
from pyod.models.iforest import IForest
from rpy2 import robjects
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process.kernels import RBF
import tensorflow_probability as tfp
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import OneClassSVM

from kernel import create_kernel
from log_regression import KLR
from visualization import plot_classification


def create_kernel_sklearn(data, gamma=1, kernel='rbf'):
    '''
    Returns K, k(_, _)
    '''
    length_scale = np.sqrt(1.0 / (2.0 * gamma))
    # linear kernel
    kernel_function = linear_kernel
    if kernel == 'rbf':
        kernel_function = RBF(length_scale)
    return kernel_function(data), kernel_function


def create_kernel_tfp(data, gamma=1, kernel='rbf'):
    '''
    Returns K, k(_, _)
    '''
    length_scale = np.sqrt(1.0 / (2.0 * gamma))
    scalar_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale)
    return scalar_kernel.matrix(data, data), scalar_kernel.apply

def create_kernel_raw(data, gamma=1, kernel='rbf'):
    '''
    Returns K, k(_, _)
    '''
    kernel_function = lambda x, y: np.exp(-gamma*np.sum(np.square(x[:, None] - y), axis=-1))
    return kernel_function(data, data), kernel_function


class KLR_R:

    def __init__(self, X, Y, kernel='rbf', C=0.01, gamma=1):
        flat_X = X.flatten('F')
        self.C = C
        self.gamma = gamma
        self.X = robjects.r['matrix'](robjects.FloatVector(flat_X), nrow=len(X))
        self.Y = robjects.FloatVector(Y)

    def predict_proba(self, X_new):
        flat_X_new = X_new.flatten('F')
        X_new_R = robjects.r['matrix'](robjects.FloatVector(flat_X_new), nrow=len(X_new))
        KLR_cb = robjects.r['KLR']
        gamma_R = robjects.FloatVector([1.0 / self.gamma])
        lambda_R = robjects.FloatVector([1.0 / self.C])
        pos_prob = np.array(KLR_cb(self.X, self.Y, X_new_R, lambda_R, kernel="exponential", power=2.0, rho=gamma_R))
        neg_prob = 1.0 - pos_prob
        return np.hstack((pos_prob[:, None], neg_prob[:, None]))

    def predict(self, x):
        predictions = self.predict_proba(x)[:, 0]
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        return predictions

class KNNC_alt:

    def __init__(self, X, y, k):
        self.model = KNeighborsClassifier(n_neighbors=min(int(k), len(X) - 1))
        # print(f'X: {X[Y == 0]} \n Y: {Y}')

        self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X):
        dist, _ = self.model.kneighbors(X)
        return dist.mean(axis=1)[:, None]

    # def predict(self, X):
    #    return self.model.predict(X)

class KNNM_alt:

    def __init__(self, X, y, k):
        self.k = k
        self.X = X
        self.y = y
        self.model = KNeighborsClassifier(n_neighbors=min(int(k), len(X) - 1))
        self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X):
        indices = self.model.kneighbors(X, return_distance=False)
        return np.linalg.norm(self.X[self.y == 0][indices].mean(axis=1) - X, axis=1)[:, None]

    def predict(self, X):
        return self.model.predict(X)

class KLR_alt:

    @staticmethod
    # @DeprecationWarning
    def _optimizing_function_R(beta, y, K, C):
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
        weights = np.reshape(beta[:-1], (1, len(y)))
        bias = beta[-1]
        regularization_factor = 0.5 * (weights @ K @ weights.T)[0, 0]
        loss_function = -y * (K @ weights.T + bias)[:, 0]  # - y*f
        classification_factor = np.sum(C * np.log(1.0 + np.exp(loss_function)))  # loss(-y*f)
        return classification_factor + regularization_factor

    @staticmethod
    @njit(fastmath=True)
    def _optimizing_function_data_independent_nopython(beta, y, K, C):
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
        #beta0 = np.random.normal(0, 1, len(y) + 1)
        beta0 = np.ones(len(y) + 1)

        res = minimize(self._optimizing_function_Paper_nopython, method="SLSQP",
                       x0=beta0, options={'disp': False}, args=(self.y, K, self.C),
                       )

        self.weights = res.x[None, :-1]
        self.bias = res.x[-1]
        return res.x

    def decision_function(self, x):
        return (self.weights @ self.kernel_function(self.X, x[:, None]) + self.bias)[0]

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


class IREOS_1:

    def __init__(self, Classifier, m_cl=1.0, C=100.0, execution_policy='parallel'):
        self.Classifier = Classifier
        self.m_cl = m_cl
        self.C_constant = C
        self.execution_policy = execution_policy
        self.X = None
        self.ground_truth = None
        self.solutions = None
        self._gamma_max = -1
        self._E_I = -1

    def classify(self, solution, gamma, C):
        if self.Classifier == KLR:
            model = KLR(kernel='rbf', gamma=gamma, C=C)
            model.fit(self.X, solution)
            # plot_classification(model, X, y)
            p = model.predict_proba(self.X[solution == 1])[:, 1]
        return p

    def get_gamma_max(self, solution):
        print(f'Getting gamma max')
        gamma = 10
        while True:
            for candidate in np.identity(len(solution))[solution >= 0.5]:
                temp_var = self.classify(candidate, gamma, self.C_constant)
                if temp_var < 0.5:
                    print(f'Got bad classification for {candidate}\n with p: {temp_var}')
                    break
            else:
                break
            print(f'going to gamma: {gamma + 10}')
            gamma += 10
        return gamma

    def thread_work(self, gamma, solution, weights):
        print(f'working... for gamma:{gamma}')
        p_gamma = []
        # TODO: probably compute faster when not full scoring evaluation
        C = self.C_constant  # * np.power(1.0 / self.m_cl, solution - candidate)
        for candidate in np.identity(len(solution)):
            # TODO: has to be improved, not necessary when self.mcl is 1.0
            C = self.C_constant  # * np.power(1.0 / self.m_cl, solution - candidate)
            p_gamma.append(self.classify(candidate, gamma, C)[0])
        avg_p_gamma = np.array(p_gamma) @ weights / np.sum(weights)
        return avg_p_gamma

    def compute_ireos_for_solution(self, solution, adjusted=False):
        weights = solution

        n_gammas = 16
        # gamma_values = np.logspace(-3, np.log10(gamma_max), n_gammas)
        gamma_values = np.linspace(0.001, self.gamma_max, n_gammas)

        if self.execution_policy == 'sequential':
            p = []
            for gamma in gamma_values:
                p.append(self.thread_work(gamma, solution, weights))
        elif self.execution_policy == 'parallel':
            from multiprocessing import cpu_count
            n_jobs = cpu_count()
            print(f'Starting with {n_jobs} jobs')
            job = [delayed(self.thread_work)(gamma, solution, weights) for gamma in gamma_values]
            p = Parallel(n_jobs=n_jobs)(job)

        ireos_score = np.reciprocal(n_gammas, dtype=float) * np.sum(p)

        print(f'ireos not adjusted: {ireos_score}')

        if adjusted:
            ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

        print(f'ireos adjusted: {ireos_score}')
        print(f'auc: {roc_auc_score(self.ground_truth, solution)}')

        fig, ax = plt.subplots(2)
        ax[1].set_ylim(0, np.max(p))
        ax[1].set_xlim(0, gamma_values[-1])
        ax[0].set_title(f'ireos_adjusted: {ireos_score}')
        ax[1].plot(gamma_values, p)
        plot_classification(self.X, solution, plot=ax[0])

        return ireos_score

    def fit(self, dataset, ground_truth, solutions):
        self.X = dataset
        self.ground_truth = ground_truth
        self.solutions = solutions
        return self.gamma_max, self.E_I

    def compute_ireos_scores(self, adjusted=False):
        for solution in self.solutions:
            yield self.compute_ireos_for_solution(solution, adjusted)

    @property
    def E_I(self):
        if self._E_I < 0:
            full_solution = np.ones((len(self.X)))
            self._E_I = self.compute_ireos_for_solution(full_solution)
        else:
            return self._E_I

    @E_I.setter
    def E_I(self, value):
        self._E_I = value

    @property
    def gamma_max(self):
        if self._gamma_max < 0 and len(self.solutions):
            max_solution = np.max(self.solutions, axis=0)
            self._gamma_max = self.get_gamma_max(max_solution)
        else:
            return self._gamma_max

    @gamma_max.setter
    def gamma_max(self, value):
        self._gamma_max = value

    def dataset_score(self):
        full_solution = np.ones((len(self.X)))
        self.gamma_max = self.get_gamma_max(full_solution)
        return self.compute_ireos_for_solution(self.ground_truth, adjusted=True)

class OutlierScoringSolution(ABC):
    def __init__(self, X, y_true, solution_evaluation_metric=roc_auc_score, scoring_solution=True,
                 normalize_scores=False, **kwargs):
        self.X = X
        self.y_true = y_true
        self.solutions = None
        self.solution_scores = None
        self._scores = None
        self.solution_evaluation_metric = solution_evaluation_metric
        self.scoring_solution = scoring_solution

        self.normalize_scores = normalize_scores

    def compute_solutions(self):
        if self.scoring_solution:
            return self._compute_solutions_scoring()
        else:
            return self._compute_solutions_binary()

    def _compute_solutions_scoring(self):
        """Returns scoring solution"""
        raise NotImplementedError()

    def _compute_solutions_binary(self):
        """Returns binary solution"""
        raise NotImplementedError()

    @property
    def scores(self):
        if self._scores is None:
            if self.solution_scores is None:
                self.compute_solutions()
                if self.normalize_scores:
                    self.solution_scores = minmax_scale(self.solution_scores, axis=1)
                    # self.solution_scores = (self.solution_scores - self.solution_scores.min()) / (self.solution_scores.max() - self.solution_scores.min())

            # self._scores = np.zeros(len(self.solution_scores))
            # for i, solution in enumerate(self.solution_scores):
            #    self._scores[i] = self.solution_evaluation_metric(self.y_true, solution)

        return self._scores

    def best_solution(self):
        best = self.scores.argmax()
        return (self.scores.max(), self.scores.argmax(), self.solution_scores[best])

    def worst_solution(self):
        worst = self.scores.argmin()
        return (self.scores.min(), self.scores.argmin(), self.solution_scores[worst])


class LOFScoringSolution(OutlierScoringSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = kwargs.get("max_neighbors", 101)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_solutions_scoring(self):
        """Fills self.solution_scores array with multiple solutions.

        Higher values correspond to higher probability of it being an outlier."""
        solution_scores = []
        for n in range(1, self.max_neighbors):
            m = loop.LocalOutlierProbability(self.X, n_neighbors=n).fit()
            solution_scores.append(m.local_outlier_probabilities.astype(float))

            # lof = LocalOutlierFactor(n_neighbors=n, contamination=self.contamination)
            # lof.fit_predict(self.X)
            # solution_score = -1 * lof.negative_outlier_factor_
            # solution_scores.append(solution_score)

        self.solution_scores = np.array(solution_scores)
        return self.solution_scores

    def _compute_solutions_binary(self):
        solution_scores = []
        for n in range(1, self.max_neighbors):
            lof = LocalOutlierFactor(n_neighbors=n)
            y_pred = (lof.fit_predict(self.X) * -1 + 1) / 2
            solution_scores.append(y_pred)
        self.solution_scores = np.array(solution_scores)




class ISOForestScoringSolution(OutlierScoringSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_solutions_scoring(self):
        solution_scores = []
        for n_est in range(50, 150, 10):
            # iso_f = IsolationForest(n_estimators=n_est, random_state=1, contamination=self.contamination)
            # iso_f.fit(self.X)
            # solution_scores.append(-1 * iso_f.score_samples(self.X))
            iso_f = IForest(n_estimators=n_est, random_state=42, contamination=self.contamination)
            iso_f.fit(self.X)
            solution_scores.append(iso_f.predict_proba(self.X)[:, 1])

        self.solution_scores = np.array(solution_scores)

    def _compute_solutions_binary(self):
        solution_scores = []
        for n_est in range(50, 150, 10):
            iso_f = IsolationForest(n_estimators=n_est, random_state=1)
            y_pred = (iso_f.fit_predict(self.X) * -1 + 1) / 2
            solution_scores.append(y_pred)
        self.solution_scores = np.array(solution_scores)


class ONEClassSVMScoringSolution(OutlierScoringSolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_solutions_scoring(self):
        solution_scores = []
        for gamma in np.logspace(-1, 1, num=10):
            svm = OneClassSVM(gamma=gamma)
            svm.fit(self.X)
            solution_scores.append(-1 * svm.score_samples(self.X))
        self.solution_scores = np.array(solution_scores)

    def _compute_solutions_binary(self):
        solution_scores = []
        for gamma in np.logspace(-1, 1, num=10):
            svm = OneClassSVM(gamma=gamma)
            y_pred = (svm.fit_predict(self.X) * -1 + 1) / 2
            solution_scores.append(y_pred)
        self.solution_scores = np.array(solution_scores)

