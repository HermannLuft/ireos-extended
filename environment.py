from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PCA_Detector

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from separability_algorithms import KLR, KNNC_w, KNNC, KNNM, SVM


class Environment:

    def __init__(self, ):
        self._anomaly_detection = []
        self._separability_algorithms = []

    @property
    def anomaly_detection(self):
        return self._anomaly_detection

    @anomaly_detection.setter
    def anomaly_detection(self, kwargs):
        # Anomaly Detection setup: algorithm to use with the hyperparameter range
        # elements structured as dict(Algorithm, hyperparameter variable, optional: interval, sampling size)
        self._anomaly_detection = [
            dict(ad_algorithm=ABOD, r_name='n_neighbors', interval=(1, min(100, kwargs['n_samples']))),
            dict(ad_algorithm=KDE, r_name='bandwidth', sampling_size=100),
            dict(ad_algorithm=COF, r_name='n_neighbors', interval=(2, min(100, kwargs['n_samples']))),
            dict(ad_algorithm=LOF, r_name='n_neighbors', interval=(1, min(100, kwargs['n_samples']))),
            dict(ad_algorithm=KNN, r_name='n_neighbors', interval=(1, min(100, kwargs['n_samples'] - 1))),
            dict(ad_algorithm=IForest, r_name='n_estimators', interval=(1, min(100, kwargs['n_samples']))),
            dict(ad_algorithm=OCSVM, r_name='nu', interval=(0.25, 0.75), sampling_size=100),
            dict(ad_algorithm=PCA_Detector, r_name='n_components',
                 interval=(1, min(kwargs['n_samples'], kwargs['n_dimensions']))),
            #dict(ad_algorithm=LODA, r_name='n_random_cuts', interval=(1, min(100, kwargs['n_samples'])),
            #     ad_kwargs={'n_bins': 'auto'}),
            dict(ad_algorithm=HBOS, r_name='n_bins', interval=(3, min(100, kwargs['n_samples']))),
        ]

    @property
    def separability_algorithms(self):
        return self._separability_algorithms

    @separability_algorithms.setter
    def separability_algorithms(self, kwargs):
        # Anomaly Detection setup: algorithm to use with the hyperparameter range
        # separability algorithms to use for the ireos index with parameter, metric, sampling and args for the
        # implemented algorithm class (e.g. SVC from scikit-learn)
        # kwargs: attributes obtained by main function, e.g. kwargs['n_samples']
        # elements structured as dict(Name, separability algorithm, arguments for IREOS)
        self._separability_algorithms = [
            ('KLR_p', KLR, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
            ('KLR_f', KLR, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
            #('SVM_p', SVM, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
            #('SVM_f', SVM_Test, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
            ('LRG_nystroem_p', LogisticRegression, dict(metric='probability', sample_size=100, kernel_leverage='Nystroem',
                                                        c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
                                                                    solver='liblinear'))),
            ('LRG_nystroem_f', LogisticRegression, dict(metric='decision', sample_size=100, kernel_leverage='Nystroem',
                                                        c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
                                                                    solver='liblinear'))),
            ('SVC_p', SVC, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100,
                                                                                                   probability=False,
                                                                                                   random_state=0))),
            ('SVC_f', SVC, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100,
                                                                                                probability=False,
                                                                                                random_state=0))),
            ('LRG_linear', LogisticRegression, dict(metric='probability', c_args=dict(C=100))),
            ('SVC_linear', LinearSVC, dict(metric='decision', c_args=dict(C=100))),
            ('KNNC_10%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * kwargs['n_samples']))),
            ('KNNC_50%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * kwargs['n_samples']))),
            ('KNNM_10%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * kwargs['n_samples']))),
            ('KNNM_50%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * kwargs['n_samples']))),
            #('KNNC_W_50%', KNNC_w, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * kwargs['n_samples'])
            #                           , solution_dependent=True)),
            ('MLP', MLPClassifier, dict(r_name='max_iter', metric='probability', balance_class=True,
                                       r_min=1, r_max=25,
                                       c_args=dict(random_state=0))),
            #('IsoForest', IForest, dict(metric='probability', c_args=dict(random_state=0)))
        ]



