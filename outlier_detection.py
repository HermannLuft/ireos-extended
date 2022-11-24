import abc

import numpy as np
from pyod.models.kde import KDE
from pyod.models.ocsvm import OCSVM
from sklearn.decomposition import PCA
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
# All algorithms are implemented in JAVA by ELKI

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
        # self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        # TODO: why not uses k ?
        # TODO: mikowski distance as default?
        model = KDE().fit(self.X)
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
        self.max_neighbors = min(kwargs.get("max_neighbors", 101), 100)
        self.contamination = kwargs.get("contamination", 0.05)

    def _compute_scores(self):
        solution_scores = []
        model = PCA_pyod().fit(self.X)
        solution_scores.append(model.predict_proba(self.X, method='unify')[:, 1])
        return np.vstack((*solution_scores,))

