import os
import sys
from bisect import bisect
from itertools import combinations

import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

sys.path.append('../../')
from ireos import IREOS_KNN, IREOS_LC

import numpy as np
from matplotlib import pyplot as plt
import numba
from numba import jit, njit, objmode
from sklearn.metrics import auc, roc_auc_score

from data import get_synthetic_features, create_data, get_parkinson_X_y, get_dataset_prepared, \
    load_campos_data
from log_regression import KLR, KNNC, KNNM
from multiprocessing import Process, Queue
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from outlier_detection import LoOP, COF, FastABOD, LDF, KNN, IsoForest, OC_SVM, LoOP, LOF, LOCI, \
    PCA_Detector
from visualization import plot_classification, plot_model

# TODO: add log levels

# if the results should always be computed when already present
force_compute = False


def compute_outlier_result(algorithm, X, y, dataset=None, **kwargs):
    model = algorithm(X, y, **kwargs)
    if dataset is not None:
        path = os.path.join('memory', dataset, algorithm.__name__ + '_solution.npy')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        try:
            next_solutions = np.load(path)
        except FileNotFoundError:
            next_solutions = model.compute_scores()
            np.save(path, next_solutions)
    else:
        next_solutions = model.compute_scores()
    return next_solutions


def filtering_solutions(solutions, ground_truth):
    evaluations = [(roc_auc_score(ground_truth, solution), solution) for solution in solutions]

    print(f'All auc scores are {[round(x[0], 3) for x in evaluations]}')

    auc_scores_values = np.array([x[0] for x in evaluations])
    evenly_spaced_auc = []
    for value in np.linspace(np.min(auc_scores_values), np.max(auc_scores_values), 10):
        index = np.argmin(np.abs(auc_scores_values - value))
        evenly_spaced_auc.append(evaluations[index])
    evaluations_10 = evenly_spaced_auc

    print(f'10 evenly spaced auc scores are {[round(x[0], 3) for x in evaluations_10]}')

    return zip(*evaluations_10)


def main():
    # Data Setup
    contamination = 0.05
    dataset = "Parkinson_withoutdupl_norm_05_v09"

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)
    assert np.max(X) <= 1 and np.min(X) >= 0, "Not normalized"
    n_samples = len(X)
    plot_classification(X, y)

    # TODO: check on duplicates!
    algorithm_setting = [
        (FastABOD, {'max_neighbors': n_samples, 'contamination': contamination}),
        (LDF, {'max_neighbors': n_samples, 'contamination': contamination}),
        (LOF, {'max_neighbors': n_samples, 'contamination': contamination}),
        (LoOP, {'max_neighbors': n_samples, 'contamination': contamination}),
        (COF, {'max_neighbors': n_samples, 'contamination': contamination}),
        (KNN, {'max_neighbors': n_samples, 'contamination': contamination}),
        (IsoForest, {'max_neighbors': n_samples, 'contamination': contamination}),
        (OC_SVM, {'max_neighbors': n_samples, 'contamination': contamination}),
        # (LOCI, {'max_neighbors': n_samples, 'contamination': contamination}), Not further used due to time reasons
        (PCA_Detector, {'max_neighbors': n_samples, 'contamination': contamination}),
    ]
    job = [delayed(compute_outlier_result)(algorithm, X=X, y=y, dataset=dataset, **kwargs)
           for algorithm, kwargs in algorithm_setting]
    solutions_list = Parallel(n_jobs=len(algorithm_setting))(job)
    solutions = np.vstack(solutions_list)
    auc_scores, omega = filtering_solutions(solutions, y)

    ireos_setting = [
        (LogisticRegression, {'metric': 'probability', 'n_gammas': 100}),
        #(KLR, {'metric': 'probability', 'n_gammas': 100}),
        (LogisticRegression, {'metric': 'distance', 'n_gammas': 100}),
        (SVC, {'metric': 'probability', 'n_gammas': 100}),
        (SVC, {'metric': 'distance', 'n_gammas': 100}),
        (KNNM, {'percent': 0.1}),
        (KNNM, {'percent': 0.5}),
        (KNNC, {'percent': 0.1}),
        (KNNC, {'percent': 0.5}),
        (LinearSVC, {'metric': 'probability', 'n_gammas': 1}),
        (LinearSVC, {'metric': 'distance', 'n_gammas': 1}),
    ]

    for Maximum_Margin_Classifier, kwargs in ireos_setting:
        print(f'Compute correlation for {Maximum_Margin_Classifier.__name__} with {kwargs}')

        name = f'{Maximum_Margin_Classifier.__name__}_'
        if Maximum_Margin_Classifier in (SVC, LogisticRegression, KLR, LinearSVC):
            Ireos = IREOS_LC(Maximum_Margin_Classifier, **kwargs)
            name += kwargs["metric"]
        elif Maximum_Margin_Classifier in (KNNC, KNNM):
            Ireos = IREOS_KNN(Maximum_Margin_Classifier, **kwargs)
            name += f'{kwargs["percent"]:.0%}'
        else:
            raise NotImplementedError

        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            results = pd.DataFrame([])
        if force_compute or name not in results.index:
            #Ireos.E_I = 0
            Ireos.fit(X, omega)
            c = list(Ireos.compute_ireos_scores())
            correlation, _ = stats.spearmanr(auc_scores, c)
            results.at[name, 'correlation'] = round(correlation, 3)
            results.to_csv(path)
        else:
            correlation = results.at[name, 'correlation']

        print(f'Correlation: {correlation}')


if __name__ == '__main__':
    main()
    exit(0)
