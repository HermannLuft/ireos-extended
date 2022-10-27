import multiprocessing as mp
import time

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from numpy import int64
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics

# from noise.ireos.alternative import KLR_alt
from log_regression import KLR, KNNC, KNNM

from visualization import plot_classification


# TODO: anstatt fit methode datensätze davor laden und nach scores bewerten
# TODO: IREOS nach Classifiern KNN und Margin trennen, bessere Oberklasse

class IREOS:
    pass


class IREOS_LC(IREOS):
    """
    Using dynamic programming:
    ---candidates---
    |...............
    |...............
    g.......p.......
    |...............
    |...............

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
            # plot_classification(model, X, y)
            p = model.predict_proba(self.X[solution == 1])[:, 1]
        elif self.Classifier == LogisticRegression:
            model = LogisticRegression(C=C, penalty='l2')
            # Information: es existiert noch RBFSampler der schneller ist aber ungenauer
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
                    print(f'Sample: {np.where(candidate == 1)[0][0]} p: {temp_var} gamma: {gamma}')
                    #gamma += self.gamma_delta
                    gamma *= 1.1
                    temp_var = self.classify(candidate, C=self.C_constant, gamma=gamma)
                self.gamma_bounds[candidate == 1] = gamma
        #print(f'Computed gamma: {np.max(self.gamma_bounds[solution > 0.5])}')
        self.metric = actual_metric
        return np.max(self.gamma_bounds[solution > 0.5])

    def fit(self, dataset, solutions=[]):
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
        # gamma_values = np.logspace(np.log10(0.00001), np.log10(self.gamma_max), self.n_run_values)
        self.compute_probability_array(gamma_values=gamma_values)
        for solution in self.solutions:
            weights = solution
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=weights)
            ireos_score = np.reciprocal(self.n_gammas, dtype=float) * np.sum(avg_p_gamma)
            if self.adjustment:
                uniform_average = np.average(self.probability_array, axis=1)
                self.E_I = np.reciprocal(self.n_gammas, dtype=float) * np.sum(uniform_average)
            ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

            # visualization
            #print(f'ireos score adjusted: {ireos_score}')
            # print(f'auc: {roc_auc_score(self.ground_truth, solution)}')
            # fig, ax = plt.subplots(2)
            # ax[1].set_ylim(0, np.max(avg_p_gamma))
            # ax[1].set_xlim(0, run_values[-1])
            # ax[0].set_title(f'ireos_adjusted: {ireos_score}')
            # ax[1].plot(run_values, avg_p_gamma)
            # plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score

    @property
    def gamma_max(self):
        if self._gamma_max < 0:
            if self.Classifier in (SVC, LogisticRegression, KLR, MLPClassifier):
                max_solution = np.max(self.solutions, axis=0)
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
        else:
            raise NotImplementedError(f'{self.Classifier.__name__} not implemented!')

        return p

    def fit(self, dataset, solutions=[]):
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
            #print(f'ireos score adjusted: {ireos_score}')
            # print(f'auc: {roc_auc_score(self.ground_truth, solution)}')
            # fig, ax = plt.subplots(2)
            # ax[1].set_ylim(0, np.max(avg_p_gamma))
            # ax[1].set_xlim(0, run_values[-1])
            # ax[0].set_title(f'ireos_adjusted: {ireos_score}')
            # ax[1].plot(run_values, avg_p_gamma)
            # plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score