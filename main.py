import os
import sys
import pandas as pd
import time
from environment import Environment
from scipy import stats

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


def compute_outlier_result(X, dataset_name, ad_algorithm, **kwargs):
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
    #evaluations = list(filter(lambda x: not (x[1] <= 0.5).all(), evaluations))
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

    env: Environment = Environment()
    env.anomaly_detection = dict(n_samples=n_samples, n_dimensions=n_dimensions)
    env.separability_algorithms = dict(n_samples=n_samples)

    # compute solutions for anomaly detection
    job = [delayed(compute_outlier_result)(X=X, dataset_name=dataset, **kwargs)
           for kwargs in env.anomaly_detection]
    solutions_list = Parallel(n_jobs=len(env.anomaly_detection))(job)
    solutions = np.vstack(solutions_list)
    # 10 solutions to compute the spearman correlation with the ireos indices
    auc_scores, omega = filtering_solutions(solutions, y, amount=10)

    # compute spearman correlations and the associated ROC-AUC score to the best solution by the indices
    for name, Maximum_Margin_Classifier, kwargs in env.separability_algorithms:
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
            index_scores = list(Ireos.compute_ireos_scores())
            end_time = time.time()

            correlation, _ = stats.spearmanr(auc_scores, index_scores)
            max_index_auc = auc_scores[np.argmax(index_scores)]

            results.at[name, 'correlation'] = round(correlation, 3)
            results.at[name, 'auc_position'] = 10 - np.argmax(index_scores)
            results.at[name, 'auc_to_maxindex'] = round(max_index_auc, 3)
            results.at[name, 'time'] = end_time - start_time
            results.to_csv(path)
        else:
            correlation = results.at[name, 'correlation']

        print(f'Correlation: {correlation}')


if __name__ == '__main__':
    main()
    exit(0)
