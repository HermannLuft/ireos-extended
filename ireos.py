import multiprocessing as mp
import time

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics

# from noise.ireos.alternative import KLR_alt
from log_regression import KLR, KNNC, KNNM

from visualization import plot_classification

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class IREOS:

    # TODO: abort run_values

    """
    Using dynamic programming:
    ---candidates---
    |...............
    |...............
    g.......p.......
    |...............
    |...............

    """

    def __init__(self, Classifier, n_run_values=100, m_cl=1.0, C=100.0, execution_policy='parallel'):
        self.Classifier = Classifier
        self.m_cl = m_cl
        self.C_constant = C
        self.execution_policy = execution_policy
        self.X = None
        self.ground_truth = None
        self.solutions = None
        self._gamma_max = -1
        self._E_I = -1
        self.gamma_delta = 0.01
        self.probability_array = None
        # standard is 32
        self.n_run_values = n_run_values
        self.KNNModel = None

    # TODO: run values ersetzen in gamma -> unabhängigkeit
    def classify(self, solution, C=100, gamma=.2, k=-1, metric='probability'):
        # TODO: davon ausgehen, dass es ein Kandidat ist
        if self.Classifier == KLR:
            model = KLR(kernel='rbf', gamma=gamma, C=C)
            model.fit(self.X, solution)
            # plot_classification(model, X, y)
            p = model.predict_proba(self.X[solution == 1])[:, 1]
        elif self.Classifier == LogisticRegression:
            model = LogisticRegression(solver='liblinear', C=C, max_iter=100_000, penalty='l1')
            # TODO: es existiert noch RBFSampler der schneller ist aber ungenauer
            f_map = Nystroem(gamma=gamma, random_state=0, n_components=len(self.X))
            X_transformed = f_map.fit_transform(self.X)
            model.fit(X_transformed, solution)
            if metric == 'probability':
                p = model.predict_proba(X_transformed[solution == 1])[:, 1]
            elif metric == 'distance':
                p = model.decision_function(X_transformed[solution == 1])
                #p = np.abs(p)
        elif self.Classifier == SVC:
            # TODO: why probability=True doesn't work?
            model = SVC(kernel='rbf', C=C, gamma=gamma, probability=False, random_state=0)
            model.fit(self.X, solution)
            if metric == 'probability':
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(self.X, solution)
                p = clf.predict_proba(self.X[solution == 1])[:, 1]
            elif metric == 'distance':
                # TODO: good correlation with multiplying by np.sign(probability - 0.5)
                p = model.decision_function(self.X[solution == 1]) #*np.sign(p - 0.5)
        elif self.Classifier == KNNC:
            # TODO: muss eigentlich nur einmal trainiert werden
            # TODO: macht vll in der Reihenfolge der Parallelität nicht viel Sinn
            self.KNNModel = KNNC(self.X, solution)
            p = self.KNNModel.predict_proba(self.X[solution == 1], k)
        elif self.Classifier == KNNM:
            #if self.KNNModel is None:
            self.KNNModel = KNNM(self.X, solution)
            p = self.KNNModel.predict_proba(self.X[solution == 1], k)
        elif self.Classifier == MLPClassifier:
            model = MLPClassifier(solver='lbfgs', alpha=gamma, hidden_layer_sizes=(10, 3), random_state=1, max_iter=1000)
            model.fit(self.X, solution)
            p = model.predict_proba(self.X[solution == 1])[:, 1]
        elif self.Classifier == LinearSVC:
            model = LinearSVC(C=C, max_iter=10000, dual=False)
            model.fit(self.X, solution)
            if metric == 'probability':
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(self.X, solution)
                p = clf.predict_proba(self.X[solution == 1])[:, 1]
            elif metric == 'distance':
                # TODO: good correlation with multiplying by np.sign(probability - 0.5)
                p = model.decision_function(self.X[solution == 1])
        else:
            raise NotImplementedError(f'{self.Classifier.__name__} not implemented!')

        # TODO: another ideas:
        # normal logREG, poly with k as degrees in SVC or LR, distance to d.b.
        # TODO: bis 4 kriegt er es gut hin
        # model = SVC(kernel='poly', C=C, degree=k, probability=True, random_state=0)
        # model.fit(self.X, solution)
        # TODO: das reicht bei 1
        # model = LogisticRegression(C=C, max_iter=10000)
        # f_map = PolynomialCountSketch(degree=k, random_state=0, n_components=100)
        # X_transformed = f_map.fit_transform(self.X)
        # model.fit(self.X, solution)
        # p = model.predict_proba(self.X[solution == 1])[:, 1]
        # TODO: MLP Classifier (ANN) als gamma entweder anzahl von layern oder
        # anzahl an gradient schritten oder alpha value für regularisierung!
        return p

    def get_gamma_max(self, solution):
        # TODO: entweder im vorhinein gamma_max bei KNNC ignorieren oder ausrechnen? Sinn?
        print(f'Getting max ...')
        gamma = self.gamma_delta
        # TODO: parallelize
        for candidate in np.identity(len(solution))[solution > 0.5]:
            temp_var = self.classify(candidate, C=self.C_constant, gamma=gamma)
            while temp_var <= 0.5:
                # TODO: in einer Zeile ausgeben
                print(f'Sample: {np.where(candidate == 1)[0][0]} p: {temp_var} gamma: {gamma}')
                gamma += self.gamma_delta
                temp_var = self.classify(candidate, C=self.C_constant, gamma=gamma)
        print(f'Computed gamma: {gamma}')
        return gamma

    def fit(self, dataset, ground_truth, solutions=[]):
        self.X = dataset
        self.ground_truth = ground_truth
        self.solutions = solutions
        self.gamma_delta = self.gamma_delta * len(dataset) / (len(dataset[0]) * len(dataset[0]))
        return self.gamma_max, self.E_I

    def compute_probability_array(self, run_values, with_sparse_solution=None, metric='probability'):
        self.probability_array = np.zeros((self.n_run_values, len(self.ground_truth)))
        for i, run_variable in enumerate(run_values):
            #print(f'Working for gamma: {run_variable}')
            from multiprocessing import cpu_count
            if with_sparse_solution is not None:
                candidates = np.identity(len(self.ground_truth))[with_sparse_solution > 0]
            else:
                candidates = np.identity(len(self.ground_truth))
            n_jobs = cpu_count()
            job = [delayed(self.classify)(candidate, C=self.C_constant, gamma=run_variable,
                                          k=run_variable, metric=metric)
                   for candidate in candidates]
            if with_sparse_solution is not None:
                self.probability_array[i, with_sparse_solution > 0, None] = Parallel(n_jobs=n_jobs)(job)
            else:
                self.probability_array[i] = Parallel(n_jobs=n_jobs)(job)

    def compute_ireos_scores(self, adjusted=False, metric='probability'):
        # TODO Am Ende soll die Linie 0 anfangen
        run_values = np.linspace(0.00001, self.gamma_max, self.n_run_values)
        if self.Classifier in (KNNC, KNNM):
            run_values = np.arange(1, self.gamma_max + 1)
        # gamma_values = np.logspace(np.log10(0.00001), np.log10(self.gamma_max), self.n_run_values)
        self.compute_probability_array(run_values=run_values, metric=metric)
        for solution in self.solutions:
            # TODO: anpassen matrix
            weights = solution
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=weights)
            # avg_p_gamma = np.array(self.probability_array) @ weights / np.sum(weights)
            ireos_score = np.reciprocal(self.n_run_values, dtype=float) * np.sum(avg_p_gamma)
            print(f'ireos score raw: {ireos_score}')
            if adjusted:
                ireos_score = (ireos_score - self.E_I) / (1 - self.E_I)

            # visualization
            print(f'ireos score adjusted: {ireos_score}')
            #print(f'auc: {roc_auc_score(self.ground_truth, solution)}')
            #fig, ax = plt.subplots(2)
            #ax[1].set_ylim(0, np.max(avg_p_gamma))
            #ax[1].set_xlim(0, run_values[-1])
            #ax[0].set_title(f'ireos_adjusted: {ireos_score}')
            #ax[1].plot(run_values, avg_p_gamma)
            #plot_classification(self.X, solution, plot=ax[0])
            yield ireos_score

    @property
    def E_I(self):
        if self._E_I < 0:
            full_solution = np.ones((len(self.X)))
            E_gamma_max = self.get_gamma_max(full_solution)
            gamma_values = np.linspace(self.gamma_delta, E_gamma_max, self.n_run_values)
            self.compute_probability_array(run_values=gamma_values)
            avg_p_gamma = np.average(self.probability_array, axis=1, weights=full_solution)
            self._E_I = np.reciprocal(self.n_run_values, dtype=float) * np.sum(avg_p_gamma)
        else:
            return self._E_I

    @E_I.setter
    def E_I(self, value):
        self._E_I = value

    @property
    def gamma_max(self):
        if self._gamma_max < 0:
            if self.Classifier in (SVC, LogisticRegression, KLR, MLPClassifier):
                max_solution = np.max(self.solutions, axis=0) if len(self.solutions) else np.ones((len(self.X)))
                self._gamma_max = self.get_gamma_max(max_solution)
            elif self.Classifier in (KNNC, KNNM):
                self._gamma_max = self.n_run_values
            elif self.Classifier in (LinearSVC,):
                self._gamma_max = 1
        return self._gamma_max

    @gamma_max.setter
    def gamma_max(self, value):
        self._gamma_max = value

    def dataset_score(self):
        # TODO: check if this is needed or only ground thruth
        # full_solution = np.ones((len(self.X)))
        # self.gamma_max = self.get_gamma_max(full_solution)
        gamma_values = np.linspace(0.01, self.gamma_max, self.n_run_values)
        self.compute_probability_array(run_values=gamma_values, with_sparse_solution=self.ground_truth)
        avg_p_gamma = np.average(self.probability_array, axis=1, weights=self.ground_truth)
        ireos_score = np.reciprocal(self.n_run_values, dtype=float) * np.sum(avg_p_gamma)
        return ireos_score
