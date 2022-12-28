import abc

import numpy as np
from pyod.models.kde import KDE
from pyod.models.ocsvm import OCSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import minmax_scale
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score
from PyNomaly import loop
from pyod.models.iforest import IForest
from pyod.models.cof import COF as COF_pyod
from pyod.models.abod import ABOD
from pyod.models.loci import LOCI as LOCI_pyod
from pyod.models.knn import KNN as KNN_pyod
from pyod.models.pca import PCA as PCA_pyod
from pyod.models.lof import LOF as LOF_pyod
from abc import ABC


# TODO: missing INFLO, KDEOS, KNNW, ODIN comparing to paper of Marques
# more infos on http://ethesis.nitrkl.ac.in/5130/1/109CS0195.pdf
# GLOSH: https://hdbscan.readthedocs.io/en/latest/outlier_detection.html

class AnomalyDetector:
    '''
    Expects PyOD compatible AD algorithms with hyperparameter range to evaluate
    '''


    def __init__(self, ad_algorithm, r_name, interval=(0, 1), sampling_size=None, ad_kwargs={}):
        self.models = None
        self.ad_algorithm = ad_algorithm
        self.r_name = r_name
        self.ad_kwargs = ad_kwargs
        self.interval = interval
        self.sampling_size = sampling_size

    def fit(self, X):
        # producing (interval_start, interval_end] as advantageous for most ad algorithms
        if not self.sampling_size:
            # discrete
            parameters = np.arange(*(self.interval[0], self.interval[1] + 1), dtype=int)
        else:
            # indiscrete
            parameters = (1 - np.linspace(*self.interval, self.sampling_size, endpoint=False))[::-1]

        self.models = [self.ad_algorithm(**{self.r_name: parameter.item()}, **self.ad_kwargs).fit(X)
                       for parameter in parameters]

        return self

    def compute_scores(self, X):
        # compute normalized AD scores by Kriegel
        solutions = [model.predict_proba(X, method='unify')[:, 1] for model in self.models]
        # filter solutions containing NaNs
        solutions = np.vstack((*solutions,))[np.isnan(solutions).any(axis=1).__invert__()]

        return solutions
