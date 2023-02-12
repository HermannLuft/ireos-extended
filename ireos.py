import abc
import multiprocessing as mp
from time import sleep
from typing import List

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from numpy import int64
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings

# from noise.ireos.alternative import KLR_alt
from separability_algorithms import KLR, KNNC, KNNM, SVM

from visualization import plot_classification
import numpy.typing as npt

plotting = False


# TODO: tqdm: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution


class IREOS:
    """
    Create IREOS class with individual separability algorithm and the parameter range

    => IREOS index for multiple solutions can be computed faster
    See formula for IREOS by Marques et al. for further investigations

    Application example:
    Object = IREOS(KLR, ...)
    Object.fit(Dataset, [*solutions])
    Index_evaluations = Object.compute_scores()

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
    solution_dependent: Outlierness depends on solution
    c_args: arguments for the classifier
    """

    def __init__(self, Classifier, r_name=None, r_min=None, r_max=None,
                 sample_size=None, adjustment=False, metric='probability',
                 kernel_leverage='linear', balance_class=False, discrete_values=False,
                 solution_dependent=False, c_args=dict()):

        self.solution_dependent = solution_dependent
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

    def estimate_r_max(self) -> None:
        if self.r_name == 'gamma' or self.kernel_leverage == 'Nystroem':
            max_solution = np.ones_like(self.solutions[0]) if self.adjustment else np.max(self.solutions, axis=0)
            self.r_max = self.get_gamma_max(max_solution)

    def get_gamma_max(self, solution: npt.ArrayLike) -> float:
        """
        Estimates gamma_max with exponential increase sequentially on every candidate
        """

        if self.metric == 'probability':
            minimum_output = 0.5
        elif self.metric == 'decision':
            # TODO: not a good dependency, maybe just set output to zero in this case
            if self.Classifier in [KLR, LogisticRegression, SVC, SVM]:
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
    def classify(self, candidate: npt.ArrayLike, r_value: float, w: npt.ArrayLike = None) -> float:
        """
        Computes the function s(x_j, v)
        """
        y = w if self.solution_dependent else candidate

        if self.r_name is not None:
            # classifying with variable v: s(x_j, v)
            model = self.Classifier(**{self.r_name: r_value}, **self.c_args)
        else:
            # classifying s(x_j)
            model = self.Classifier(**self.c_args)

        # Dimension leverage of X
        if self.kernel_leverage == 'linear':
            X_transformed = self.X
        elif self.kernel_leverage == 'Nystroem':
            f_map = Nystroem(gamma=r_value, random_state=0, n_components=len(self.X))
            X_transformed = f_map.fit_transform(self.X)
        else:
            raise NotImplementedError(f'Kernel {self.kernel_leverage} not implemented')

        # Oversampling candidate
        if self.balance_class:
            more_samples = np.repeat(X_transformed[candidate == 1], len(candidate == 0) - 2, axis=0)
            X_balanced = np.vstack((more_samples, X_transformed))
            Y_balanced = np.hstack((np.ones(len(more_samples)), y))
        else:
            X_balanced = X_transformed
            Y_balanced = y

        # TODO: kNN's doesn't really fit here
        model.fit(X_balanced, Y_balanced)

        if self.metric == 'probability':
            # s(x_j, ...) = p(x_j, ...)
            if not hasattr(model, 'predict_proba'):
                #print(f'candidate: {candidate}')
                #print(f'model prediction: {model.decision_function(X_balanced)}')
                clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
                clf.fit(X_transformed, candidate)
            else:
                clf = model
            p = clf.predict_proba(X_transformed[candidate == 1])[:, 1]
        elif self.metric == 'decision':
            # s(x_j, ...) = f(x_j, ...)
            p = model.decision_function(X_transformed[candidate == 1])

        return p

    def fit(self, data, solutions=[]):
        """
        Defines parameter range
        """

        assert len(solutions), "Solutions have to be passed to fit function"
        self.X = data
        self.solutions = solutions

        if self.r_name is not None or self.r_name is None and self.kernel_leverage != 'linear':
            # Computes parameter range for varying parameters: KLR, SVM, KNN, MLP ...
            if self.r_min is None:
                self.r_min = 0.01 * len(self.X) / (len(self.X[0]) * len(self.X[0]))
            if self.r_max is None:
                self.estimate_r_max()
            if self.sample_size is not None:
                if self.discrete_values:
                    self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size, endpoint=False).astype(int)
                else:
                    self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size + 1)[1:]
                    #self.r_values = (self.r_max - np.linspace(0, self.r_max, self.sample_size, endpoint=False))[::-1]
                    # self.r_values = np.linspace(self.r_min, self.r_max, self.sample_size)

            else:
                self.r_values = np.arange(1, self.r_max + 1)
        else:
            # Computes parameter range for single parameter value: Linear Classifiers
            self.r_values = np.array([-1])

    def compute_probability_array(self, w: np.ndarray = None, with_sparse_solution=None) -> None:
        """
        Computes probabilities using dynamic programming: s(x, v) for all x and v
        ---candidates---
        |...............
        |...............
        v.......s.......
        |...............
        |...............
        ________________
        Example KLR: s(x, v) = p(x, gamma), KNN: s(x, v) = d(x, k)

        with_sparse solution:
        """
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
            job = [delayed(self.classify)(candidate, r_value, w)
                   for candidate in candidates]
            if with_sparse_solution is not None:
                self.probability_array[i, with_sparse_solution > 0.5, None] = Parallel(n_jobs=n_jobs)(job)
            else:
                self.probability_array[i] = Parallel(n_jobs=n_jobs)(job)
        print()

    def compute_ireos_scores(self) -> List[float]:
        """
        Computes I(w) for solutions
        """

        if not self.solution_dependent:
            # computes s(x_j, v) for each candidate and v
            self.compute_probability_array()
        for solution in self.solutions:
            if self.solution_dependent:
                # computes s(x_j, v, w) with m_cl or for KNN_W
                self.compute_probability_array(solution)
            # TODO: Replace by numpy functions
            if self.adjustment:
                # computes E_I
                uniform_average = np.average(self.probability_array, axis=1)
                self.E_I = np.reciprocal(len(self.r_values), dtype=float) * np.sum(uniform_average)
            weights = solution

            # Computes I after Marques et al
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
    def gamma_max(self) -> float:
        if self._gamma_max < 0:
            # computes gamma_max if not already precomputed
            if self.Classifier in (SVC, LogisticRegression, KLR, MLPClassifier):
                max_solution = np.ones_like(self.solutions[0]) if self.adjustment else np.max(self.solutions, axis=0)
                self._gamma_max = self.get_gamma_max(max_solution)
            elif self.Classifier in (LinearSVC,):
                self._gamma_max = 1
        return self._gamma_max

    @gamma_max.setter
    def gamma_max(self, value: float):
        self._gamma_max = value
