import abc
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
from separability_algorithms import KLR
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


class IREOS:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def classify(self):
        pass

    @property
    @abc.abstractmethod
    def fit(self):
        pass

    @property
    @abc.abstractmethod
    def compute_ireos_scores(self):
        pass


class IREOS_LC(IREOS):
    # TODO: gamma_delta entfernen
    """
    Using dynamic programming:
    ---candidates---
    |...............
    |...............
    g.......p.......
    |...............
    |...............
    => Ireos Index for multiple solutions can be computed faster
    """

    def __init__(self, Classifier, n_gammas=100, m_cl=1.0, C=100.0, adjustment=False, metric='probability'):
        self.gamma_bounds = None
        self.E_I = 0
        self.Classifier = Classifier
        self.m_cl = m_cl
        self.C_constant = C
        self.X = None
        self.solutions = None
        self._gamma_max = -1
        self.metric = metric
        self.gamma_delta = 0.01
        self.probability_array = None
        self.adjustment = adjustment
        self.n_gammas = n_gammas

    def classify(self, solution, C=100, gamma=.2):
        if self.Classifier == KLR:
            model = KLR(kernel='rbf', gamma=gamma, C=C)
            model.fit(self.X, solution)
            if self.metric == 'probability':
                p = model.predict_proba(self.X[solution == 1])[:, 1]
            elif self.metric == 'distance':
                p = model.decision_function(self.X[solution == 1])
        elif self.Classifier == LogisticRegression:
            model = LogisticRegression(C=C, penalty='l2')
            # Information: RBFSampler can also be used
            f_map = Nystroem(gamma=gamma, random_state=0, n_components=len(self.X))
            X_transformed = f_map.fit_transform(self.X)
            model.fit(X_transformed, solution)
            if self.metric == 'probability':
                p = model.predict_proba(X_transformed[solution == 1])[:, 1]
            elif self.metric == 'distance':
                p = model.decision_function(X_transformed[solution == 1])
        elif self.Classifier == SVC:
            model = SVC(kernel='rbf', C=C, gamma=gamma, probability=False, random_state=0)
            model.fit(self.X, solution)
            if self.metric == 'probability':
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(self.X, solution)
                p = clf.predict_proba(self.X[solution == 1])[:, 1]
            elif self.metric == 'distance':
                p = model.decision_function(self.X[solution == 1])
        elif self.Classifier == LinearSVC:
            model = LinearSVC(C=C, max_iter=10000, dual=False)
            model.fit(self.X, solution)
            if self.metric == 'probability':
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(self.X, solution)
                p = clf.predict_proba(self.X[solution == 1])[:, 1]
            elif self.metric == 'distance':
                p = model.decision_function(self.X[solution == 1])
        else:
            raise NotImplementedError(f'{self.Classifier.__name__} not implemented!')

        return p

    def get_gamma_max(self, solution):
        print(f'Getting max ...')
        actual_metric = self.metric
        self.metric = 'probability'
        gamma = self.gamma_delta
        for candidate in np.identity(len(solution))[solution > 0.5]:
            if self.gamma_bounds[candidate == 1] == 0:
                temp_var = self.classify(candidate, C=self.C_constant, gamma=gamma)
                while temp_var <= 0.5:
                    print(f'\rSample: {np.where(candidate == 1)[0][0]} p: {temp_var} gamma: {gamma}', end="")
                    # gamma += self.gamma_delta
                    gamma *= 1.1
                    temp_var = self.classify(candidate, C=self.C_constant, gamma=gamma)
                self.gamma_bounds[candidate == 1] = gamma
        # print(f'Computed gamma: {np.max(self.gamma_bounds[solution > 0.5])}')
        self.metric = actual_metric
        return np.max(self.gamma_bounds[solution > 0.5])

    def fit(self, dataset, solutions=[]):
        assert len(solutions), "Solutions have to be passed to fit function"
        self.X = dataset
        self.solutions = solutions
        self.gamma_bounds = np.zeros((len(self.X)))
        self.gamma_delta = self.gamma_delta * len(dataset) / (len(dataset[0]) * len(dataset[0]))
        return self.gamma_max

    def compute_probability_array(self, gamma_values, with_sparse_solution=None):
        self.probability_array = np.zeros((self.n_gammas, len(self.X)))
        for i, gamma in enumerate(gamma_values):
            # print(f'Working for gamma: {run_variable}')
            from multiprocessing import cpu_count
            if with_sparse_solution is not None:
                candidates = np.identity(len(self.X))[with_sparse_solution > 0.5]
            else:
                candidates = np.identity(len(self.X))
            n_jobs = cpu_count()
            job = [delayed(self.classify)(candidate, C=self.C_constant, gamma=gamma)
                   for candidate in candidates]
            if with_sparse_solution is not None:
                self.probability_array[i, with_sparse_solution > 0.5, None] = Parallel(n_jobs=n_jobs)(job)
            else:
                self.probability_array[i] = Parallel(n_jobs=n_jobs)(job)

    def compute_ireos_scores(self):
        gamma_values = np.linspace(self.gamma_delta, self.gamma_max, self.n_gammas + 1)[1:]
        # Log-Scale:
        # gamma_values = np.logspace(np.log10(0.00001), np.log10(self.gamma_max), self.n_run_values)
        self.compute_probability_array(gamma_values=gamma_values)
        if self.adjustment:
            uniform_average = np.average(self.probability_array, axis=1)
            self.E_I = np.reciprocal(self.n_gammas, dtype=float) * np.sum(uniform_average)
        # TODO: Replace by numpy functions
        for solution in self.solutions:
            weights = solution
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=weights)
            ireos_score = np.reciprocal(self.n_gammas, dtype=float) * np.sum(avg_p_gamma)
            ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

            # visualization
            if plotting:
                fig, ax = plt.subplots(2)
                ax[1].set_ylim(0, np.max(avg_p_gamma))
                ax[1].set_xlim(0, gamma_values[-1])
                ax[0].set_title(f'ireos_adjusted: {ireos_score}')
                ax[1].plot(gamma_values, avg_p_gamma)
                plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score

    @property
    def gamma_max(self):
        if self._gamma_max < 0:
            if self.Classifier in (SVC, LogisticRegression, KLR, MLPClassifier):
                max_solution = np.ones_like(self.solutions[0]) if self.adjustment else np.max(self.solutions, axis=0)
                self._gamma_max = self.get_gamma_max(max_solution)
            elif self.Classifier in (LinearSVC,):
                self._gamma_max = 1
        return self._gamma_max

    @gamma_max.setter
    def gamma_max(self, value):
        self._gamma_max = value


class IREOS_KNN(IREOS):
    """
    Using dynamic programming:
    ---candidates---
    |...............
    |...............
    g.......p.......
    |...............
    |...............

    """

    def __init__(self, Classifier, percent=0.1, m_cl=1.0, adjustment=False, execution_policy='parallel'):
        self.n_k = None
        self.percent = percent
        self.E_I = 0
        self.Classifier = Classifier
        self.m_cl = m_cl
        self.execution_policy = execution_policy
        self.X = None
        self.solutions = None
        self.probability_array = None
        self.adjustment = adjustment
        self.KNNModel = None

    @ignore_warnings(category=ConvergenceWarning)
    def classify(self, solution, k):
        if self.Classifier == KNNC:
            # TODO: muss eigentlich nur einmal trainiert werden
            # TODO: macht vll in der Reihenfolge der Parallelität nicht viel Sinn
            self.KNNModel = KNNC(self.X, solution)
            p = self.KNNModel.predict_proba(self.X[solution == 1], k)
        elif self.Classifier == KNNM:
            # if self.KNNModel is None:
            self.KNNModel = KNNM(self.X, solution)
            p = self.KNNModel.predict_proba(self.X[solution == 1], k)
        elif self.Classifier == MLPClassifier:
            # if self.KNNModel is None:
            more_samples = np.repeat(self.X[solution == 1], len(solution == 0) - 2, axis=0)
            X_balanced = np.vstack((more_samples, self.X))
            Y_balanced = np.hstack((np.ones(len(more_samples)), solution))
            model = MLPClassifier(random_state=42, max_iter=k * 3 + 1).fit(X_balanced, Y_balanced)
            p = model.predict_proba(self.X[solution == 1])[:, 1]
        else:
            raise NotImplementedError(f'{self.Classifier.__name__} not implemented!')

        return p

    def fit(self, dataset, solutions=[]):
        assert len(solutions), "Solutions have to be passed to fit function"
        self.X = dataset
        self.n_k = int(self.percent * len(self.X))
        self.solutions = solutions

    def compute_probability_array(self, k_values, with_sparse_solution=None, metric='probability'):
        self.probability_array = np.zeros((self.n_k, len(self.X)))
        for i, k in enumerate(k_values):
            # print(f'Working for gamma: {run_variable}')
            from multiprocessing import cpu_count
            if with_sparse_solution is not None:
                candidates = np.identity(len(self.X))[with_sparse_solution > 0.5]
            else:
                candidates = np.identity(len(self.X))
            n_jobs = cpu_count()
            job = [delayed(self.classify)(candidate, k=k)
                   for candidate in candidates]
            if with_sparse_solution is not None:
                self.probability_array[i, with_sparse_solution > 0.5, None] = Parallel(n_jobs=n_jobs)(job)
            else:
                self.probability_array[i] = Parallel(n_jobs=n_jobs)(job)

    def compute_ireos_scores(self):
        k_values = np.arange(1, self.n_k + 1)
        self.compute_probability_array(k_values=k_values)
        for solution in self.solutions:
            weights = solution
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=weights)
            ireos_score = np.reciprocal(self.n_k, dtype=float) * np.sum(avg_p_gamma)
            if self.adjustment:
                uniform_average = np.average(self.probability_array, axis=1)
                self.E_I = np.reciprocal(self.n_k, dtype=float) * np.sum(uniform_average)
            ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

            # visualization
            if plotting:
                fig, ax = plt.subplots(2)
                ax[1].set_ylim(0, np.max(avg_p_gamma))
                ax[1].set_xlim(0, k_values[-1])
                ax[0].set_title(f'ireos_adjusted: {ireos_score}')
                ax[1].plot(k_values, avg_p_gamma)
                plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score


class KNNC:

    def __init__(self, X, y):
        self.X = X
        self.kdt = KDTree(X, metric='euclidean')

        # self.model.fit(X=X[y == 0], y=y[y == 0])

    def predict_proba(self, X, k):
        # dist, _ = self.model.kneighbors(X)
        dist, _ = self.kdt.query(X, k=min(k, len(self.X) - 1) + 1, return_distance=True)
        self.k = k
        return dist[0, [[-1]]]
        # return dist[:, 1:].mean(axis=1)[:, None]

    def get_k(self):
        return self.k

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


class OutlierDetector(ABC):
    __metaclass__ = abc.ABCMeta

    SOLUTION_TYPES = ["binary", "non_binary_top_n", "scoring"]

    @property
    @abc.abstractmethod
    def solution_type(self):
        pass

    def __init__(self, X, **kwargs):
        self.X = X

    def compute_scores(self):
        return self._compute_scores()



class LOF(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        # TODO: Compare with LoOP algorithm
        solution_scores = []
        for n in range(1, self.max_neighbors):
            model = LOF_pyod(n_neighbors=n).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class LoOP(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(1, self.max_neighbors):
            m = loop.LocalOutlierProbability(self.X, n_neighbors=n).fit()
            solution_scores.append(m.local_outlier_probabilities.astype(float))
        return np.vstack((*solution_scores,))


class COF(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(2, self.max_neighbors):
            # TODO: check versions fast or memory
            model = COF_pyod(n_neighbors=n).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class FastABOD(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(3, self.max_neighbors):
            model = ABOD(n_neighbors=n, method='fast').fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class LDF(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        # TODO: why not uses k ?
        # TODO: mikowski distance as default?
        for n in range(4, self.max_neighbors):
            model = KDE(bandwidth=n / self.max_neighbors).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class KNN(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(2, self.max_neighbors - 1):
            model = KNN_pyod(n_neighbors=n).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class IsoForest(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(2, self.max_neighbors - 1):
            model = IForest(n_estimators=n).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class OC_SVM(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        # TODO:  how to interpret multiple k's ?
        for n in range(2, self.max_neighbors - 1):
            model = OCSVM(nu=n / self.max_neighbors).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class LOCI(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(2, self.max_neighbors - 1):
            model = LOCI_pyod(k=n).fit(self.X)
            solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))


class PCA_Detector(OutlierDetector):

    @property
    def solution_type(self):
        return super().SOLUTION_TYPES[2]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions = kwargs.get("dimensions", 3)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        for n in range(2, self.dimensions):
            model = PCA_pyod(n_components=n).fit(self.X)
        solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))



class IREOS:
    """
    Create IREOS class with individual separability algorithm and the parameter range
    Using dynamic programming:
    ---candidates---
    |...............
    |...............
    g.......p.......
    |...............
    |...............
    => Ireos Index for multiple solutions can be computed faster
    See formula for IREOS by Marques et al. for further investigations

    Classifier: separability algorithm for the IREOS index (must have fit function)
    r_name optional: value range of hyperparameter, e.g. gamma for KLR in the original IREOS
    r_min optional: hyperparameter min
    r_max optional: hyperparameter max
    sample_size optional: How many samples of separability curve
    adjustment: adjustment for chance
    metric: 'probability' for p and decision for 'f'
    kernel_leverage: leverage into higher dimensions of input data
    balance_class: oversample outlier
    discrete_values: forces hyperparameter to be only integers
    c_args: arguments for the classifier
    """

    def __init__(self, Classifier, r_name=None, r_min=None, r_max=None,
                 sample_size=None, adjustment=False, metric='probability',
                 kernel_leverage='linear', balance_class=False, discrete_values=False,
                 c_args=dict()):

        self.r_name = r_name
        self.kernel_leverage = kernel_leverage
        self.c_args = c_args
        self.sample_size = sample_size
        self.r_min = r_min
        self.E_I = 0
        self.discrete_values = discrete_values
        self.Classifier = Classifier
        self._gamma_max = -1
        self.metric = metric
        self.adjustment = adjustment
        self.balance_class = balance_class
        self.r_max = r_max
        self.r_values = None
        self.probability_array = None
        self.solutions = None
        self.X = None
        self.run_values = None

    def estimate_r_max(self):
        if self.r_name == 'gamma' or self.kernel_leverage == 'Nystroem':
            max_solution = np.ones_like(self.solutions[0]) if self.adjustment else np.max(self.solutions, axis=0)
            self.r_max = self.get_gamma_max(max_solution)

    def get_gamma_max(self, solution):

        if self.metric == 'probability':
            minimum_output = 0.5
        elif self.metric == 'decision':
            if self.Classifier in [KLR, LogisticRegression, SVC]:
                minimum_output = 0
        else:
            raise NotImplementedError(f'Metric {self.metric} not implemented!')

        print(f'Getting \u03B3 max for minimum value: {minimum_output}')
        gamma = 0.01 * len(self.X) / (len(self.X[0]) * len(self.X[0]))
        for candidate in np.identity(len(solution))[solution > 0.5]:
            temp_var = self.classify(candidate, r_value=gamma)
            while temp_var < minimum_output:
                # print(type(temp_var[0]))
                print(f'\rSample: {np.where(candidate == 1)[0][0]} p: {temp_var[0]:.2f} \u03B3: {gamma:.2f}', end="")
                # print(f'Sample: {np.where(candidate == 1)[0][0]} p: {temp_var} \u03B3: {gamma}', end="\n")
                gamma *= 1.1
                temp_var = self.classify(candidate, r_value=gamma)

        print(f'\nComputed \u03B3 max: {gamma:.2f}')
        return gamma

    @ignore_warnings(category=ConvergenceWarning)
    def classify(self, solution, r_value):
        if self.r_name is not None:
            model = self.Classifier(**{self.r_name: r_value}, **self.c_args)
        else:
            model = self.Classifier(**self.c_args)

        if self.kernel_leverage == 'linear':
            X_transformed = self.X
        elif self.kernel_leverage == 'Nystroem':
            f_map = Nystroem(gamma=r_value, random_state=0, n_components=len(self.X))
            X_transformed = f_map.fit_transform(self.X)
        else:
            raise NotImplementedError(f'Kernel {self.kernel_leverage} not implemented')

        if self.balance_class:
            more_samples = np.repeat(X_transformed[solution == 1], len(solution == 0) - 2, axis=0)
            X_balanced = np.vstack((more_samples, X_transformed))
            Y_balanced = np.hstack((np.ones(len(more_samples)), solution))
        else:
            X_balanced = X_transformed
            Y_balanced = solution

        # TODO: kNN's doesn't really fit here
        model.fit(X_balanced, Y_balanced)

        if self.metric == 'probability':
            if not hasattr(model, 'predict_proba'):
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(X_transformed, solution)
            else:
                clf = model
            p = clf.predict_proba(X_transformed[solution == 1])[:, 1]
        elif self.metric == 'decision':
            p = model.decision_function(X_transformed[solution == 1])

        return p

    def fit(self, data, solutions=[]):
        assert len(solutions), "Solutions have to be passed to fit function"
        self.X = data
        self.solutions = solutions
        if self.r_name is not None or self.r_name is None and self.kernel_leverage != 'linear':
            if self.r_min is None:
                self.r_min = 0.01 * len(self.X) / (len(self.X[0]) * len(self.X[0]))
            if self.r_max is None:
                self.estimate_r_max()
            if self.sample_size is not None:
                if self.discrete_values:
                    self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size, endpoint=False).astype(int)
                else:
                    self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size + 1)[1:]
                    # self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size)

            else:
                self.r_values = np.arange(1, self.r_max + 1)
        else:
            self.r_values = np.array([-1])

    def compute_probability_array(self, with_sparse_solution=None):
        self.probability_array = np.zeros((len(self.r_values), len(self.X)))
        for i, r_value in enumerate(self.r_values):
            if len(self.r_values) > 1:
                print(f'\rWorking for \u03B3 level : [{r_value:.2f}/{self.r_max:.2f}]', end="")
            else:
                print(f'\rWorking for one classification', end="")
            from multiprocessing import cpu_count
            if with_sparse_solution is not None:
                candidates = np.identity(len(self.X))[with_sparse_solution > 0.5]
            else:
                candidates = np.identity(len(self.X))
            n_jobs = cpu_count()
            job = [delayed(self.classify)(candidate, r_value)
                   for candidate in candidates]
            if with_sparse_solution is not None:
                self.probability_array[i, with_sparse_solution > 0.5, None] = Parallel(n_jobs=n_jobs)(job)
            else:
                self.probability_array[i] = Parallel(n_jobs=n_jobs)(job)
        print()

    def compute_ireos_scores(self):
        # Log-Scale:
        # gamma_values = np.logspace(np.log10(0.00001), np.log10(self.gamma_max), self.n_run_values)
        self.compute_probability_array()
        if self.adjustment:
            uniform_average = np.average(self.probability_array, axis=1)
            self.E_I = np.reciprocal(len(self.r_values), dtype=float) * np.sum(uniform_average)
        # TODO: Replace by numpy functions
        for solution in self.solutions:
            self.compute_probability_array()
            weights = solution
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=weights)
            ireos_score = np.reciprocal(len(self.r_values), dtype=float) * np.sum(avg_p_gamma)
            ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

            # visualization
            if plotting:
                fig, ax = plt.subplots(2)
                ax[1].set_ylim(0, np.max(avg_p_gamma))
                ax[1].set_xlim(0, self.r_values[-1])
                ax[0].set_title(f'ireos_adjusted: {ireos_score}')
                ax[1].plot(self.r_values, avg_p_gamma)
                plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score

    @property
    def gamma_max(self):
        if self._gamma_max < 0:
            if self.Classifier in (SVC, LogisticRegression, KLR, MLPClassifier):
                max_solution = np.ones_like(self.solutions[0]) if self.adjustment else np.max(self.solutions, axis=0)
                self._gamma_max = self.get_gamma_max(max_solution)
            elif self.Classifier in (LinearSVC,):
                self._gamma_max = 1
        return self._gamma_max

    @gamma_max.setter
    def gamma_max(self, value):
        self._gamma_max = value
