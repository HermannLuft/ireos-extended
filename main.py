import os
import sys
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import time
from pyod.models.pca import PCA as PCA_Detector
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from separability_algorithms import KLR, KNNC_w
from ireos import IREOS

import numpy as np
from matplotlib import pyplot as plt
import numba
from numba import jit, njit, objmode
from sklearn.metrics import auc, roc_auc_score

from data import get_synthetic_features, create_data, get_parkinson_X_y, get_dataset_prepared, \
    load_campos_data
# from log_regression import KLR
from separability_algorithms import KNNM, KNNC
from joblib import Parallel, delayed

from outlier_detection import AnomalyDetector
from visualization import plot_classification, plot_model

"""
This module computes the Spearman correlation and some additional facts
on a dataset by predefined separability algorithms and
anomaly detection models.
Dataset must be placed at ./datasets/prefix/!
input: datasetname.arff 
"""

# if the correlations should always be computed even when already present
force_compute = True


def compute_outlier_result(X, y, dataset_name, ad_algorithm, **kwargs):
    """
    Computes outlier detection results for ad_algorithm on X
    dataset_name: ID to memorize result
    kwargs: arguments for the AnomalyDetector class
    """

    detector = AnomalyDetector(ad_algorithm, **kwargs)
    if dataset_name is not None:
        path = os.path.join('memory', dataset_name, ad_algorithm.__name__ + '_solution.npy')
        # TODO: THREAD race condition possible
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        try:
            next_solutions = np.load(path)
        except FileNotFoundError:
            next_solutions = detector.fit(X).compute_scores(X)
            np.save(path, next_solutions)
    else:
        next_solutions = detector.fit(X).compute_scores(X)

    return next_solutions


def filtering_solutions(solutions, ground_truth, amount=10):
    """
    Extracts 10 evenly spaced solutions by ROC-AUC score with ground_truth
    including minimum and maximum
    """

    evaluations = [(roc_auc_score(ground_truth, solution), solution) for solution in solutions]
    # filter solutions with equal score for every sample
    evaluations = list(filter(lambda x: not (x[1] == x[1][0]).all(), evaluations))
    # print(f'All auc scores are {[round(x[0], 3) for x in evaluations]}')

    auc_scores_values = np.array([x[0] for x in evaluations])
    evenly_spaced_auc = []
    # create linspace interval and extract unique solutions to the closest ROC-AUC
    for value in np.linspace(np.min(auc_scores_values), np.max(auc_scores_values), amount):
        index = np.argmin(np.abs(auc_scores_values - value))
        evenly_spaced_auc.append(evaluations[index])
        del evaluations[index]
        auc_scores_values = np.array([x[0] for x in evaluations])
    evaluations_10 = evenly_spaced_auc

    print(f'{amount} evenly spaced auc scores are {[round(x[0], 3) for x in evaluations_10]}')

    return zip(*evaluations_10)


def main():
    """
    Computes Spearman Correlations for separability algorithms
    Uses dataset as first argument
    Dataset must be located at ./datasets/
    """

    # load dataset as X, y
    dataset = "Arrhythmia_withoutdupl_05_v01"
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)
    assert np.max(X) <= 1 and np.min(X) >= 0, "Not normalized"
    n_samples = len(X)
    n_dimensions = len(X[0])
    plot_classification(X, y)

    # Anomaly Detection setup: algorithm to use with the hyperparameter range
    algorithm_setting = [
        dict(ad_algorithm=ABOD, r_name='n_neighbors', interval=(1, min(100, n_samples))),
        dict(ad_algorithm=KDE, r_name='bandwidth', sampling_size=100),
        dict(ad_algorithm=COF, r_name='n_neighbors', interval=(2, min(100, n_samples))),
        dict(ad_algorithm=LOF, r_name='n_neighbors', interval=(1, min(100, n_samples))),
        dict(ad_algorithm=KNN, r_name='n_neighbors', interval=(1, min(100, n_samples - 1))),
        dict(ad_algorithm=IForest, r_name='n_estimators', interval=(1, min(100, n_samples))),
        dict(ad_algorithm=OCSVM, r_name='nu', interval=(0.25, 0.75), sampling_size=100),
        dict(ad_algorithm=PCA_Detector, r_name='n_components', interval=(1, min(n_samples, n_dimensions))),
        #dict(ad_algorithm=LODA, r_name='n_random_cuts', interval=(1, min(100, n_samples)),
        #     ad_kwargs={'n_bins': 'auto'}),
        dict(ad_algorithm=HBOS, r_name='n_histograms', interval=(3, min(100, n_samples))),
    ]

    # compute solutions
    job = [delayed(compute_outlier_result)(X=X, y=y, dataset_name=dataset, **kwargs)
           for kwargs in algorithm_setting]
    solutions_list = Parallel(n_jobs=len(algorithm_setting))(job)
    solutions = np.vstack(solutions_list)
    # 10 solutions to compute the spearman correlation with the ireos indices
    auc_scores, omega = filtering_solutions(solutions, y, amount=10)

    # separability algorithms to use for the ireos index with parameter, metric, sampling and args for the
    # implemented algorithm class (e.g. SVC from scikit-learn)
    ireos_setting = [
        #('KLR_p', KLR, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
        ('KLR_f', KLR, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
        #('LRG_nystroem_p', LogisticRegression, dict(metric='probability', sample_size=100, kernel_leverage='Nystroem',
        #                                            c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
        #                                                        solver='liblinear'))),
        #('LRG_nystroem_f', LogisticRegression, dict(metric='decision', sample_size=100, kernel_leverage='Nystroem',
        #                                            c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
        #                                                        solver='liblinear'))),
        #('SVM_p', SVC, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100,
        #                                                                                       probability=False,
        #                                                                                       random_state=0))),
        #('SVM_f', SVC, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100,
        #                                                                                    probability=False,
        #                                                                                    random_state=0))),
        #('LRG_linear', LogisticRegression, dict(metric='probability', c_args=dict(C=100))),
        #('SVM_linear', LinearSVC, dict(metric='decision', c_args=dict(C=100))),
        #('KNNC_10%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * n_samples))),
        #('KNNC_50%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * n_samples))),
        #('KNNM_10%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * n_samples))),
        #('KNNM_50%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * n_samples))),
        #('KNNC_W_50%', KNNC_w, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * n_samples)
        #                            , solution_dependent=True)),
        #('MLP', MLPClassifier, dict(r_name='max_iter', metric='probability', balance_class=True,
        #                            r_min=1, r_max=25,
        #                            c_args=dict(random_state=0))),
        #('IsoForest', IForest, dict(metric='probability', c_args=dict(random_state=0)))
    ]

    # compute spearman correlations and the associated ROC-AUC score to the best solution by the indices
    for name, Maximum_Margin_Classifier, kwargs in ireos_setting:
        print(f'Compute correlation for {Maximum_Margin_Classifier.__name__} with {kwargs}')
        Ireos = IREOS(Maximum_Margin_Classifier, **kwargs)

        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            results = pd.DataFrame([])

        if force_compute or name not in results.index:
            start_time = time.time()
            Ireos.fit(X, omega)
            c = list(Ireos.compute_ireos_scores())
            end_time = time.time()
            correlation, _ = stats.spearmanr(auc_scores, c)
            max_index_auc = auc_scores[np.argmax(c)]
            results.at[name, 'correlation'] = round(correlation, 3)
            results.at[name, 'auc_position'] = 10 - np.argmax(c)
            results.at[name, 'auc_to_maxindex'] = round(max_index_auc, 3)
            results.at[name, 'time'] = end_time - start_time
            results.to_csv(path)
        else:
            correlation = results.at[name, 'correlation']

        print(f'Correlation: {correlation}')


if __name__ == '__main__':
    main()
    exit(0)
