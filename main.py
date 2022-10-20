import os
import sys
from bisect import bisect

import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

sys.path.append('../../')
from ireos import IREOS

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
force_compute = True


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


# TODO: change to compute_spearman for solutions -> pre filtering solutions
def compute_spearman(Maximum_Margin_Classifier, data, ground_truth, solutions,
                     C=100, m_cl=1, n_run_values=100, metric='probability'):
    assert data.min() >= 0 and data.max() <= 1, "Data not normalized"

    Ireos = IREOS(Maximum_Margin_Classifier, n_run_values=n_run_values, C=C)

    evaluations = [(roc_auc_score(ground_truth, solution), solution) for solution in solutions]

    print(f'All auc scores are {[round(x[0], 3) for x in evaluations]}')

    # TODO: best method to extract evenly spaced values?
    auc_scores_values = np.array([x[0] for x in evaluations])
    evenly_spaced_auc = []
    for value in np.linspace(np.min(auc_scores_values), np.max(auc_scores_values), 10):
        index = np.argmin(np.abs(auc_scores_values - value))
        evenly_spaced_auc.append(evaluations[index])
    evaluations_10 = evenly_spaced_auc

    print(f'10 evenly spaced auc scores are {[round(x[0], 3) for x in evaluations_10]}')

    auc_scores, omega = zip(*evaluations_10)
    Ireos.E_I = 0
    Ireos.fit(data, ground_truth, omega)
    c = list(Ireos.compute_ireos_scores(adjusted=True, metric=metric))
    correlation, _ = stats.spearmanr(auc_scores, c)
    return correlation


def main():
    # Data Setup
    contamination = 0.05
    # dataset = "WBC_withoutdupl_norm_v01"
    dataset = "Parkinson_withoutdupl_norm_05_v02"
    # dataset = "Wilt_withoutdupl_norm_05"
    # dataset = "Hepatitis_withoutdupl_norm_05_v01"
    # X, y = get_synthetic_features(2, 3, 100, contamination)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)
    # X, y = np.array([[0, 0], [0, 0], [0.5, 0.5]]), np.array([0, 0, 1])

    # TODO: check if normalized
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)
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
        (LOCI, {'max_neighbors': n_samples, 'contamination': contamination}),
        (PCA_Detector, {'max_neighbors': n_samples, 'contamination': contamination}),
    ]
    job = [delayed(compute_outlier_result)(algorithm, X=X, y=y, dataset=dataset, **kwargs)
           for algorithm, kwargs in algorithm_setting]
    solutions_list = Parallel(n_jobs=len(algorithm_setting))(job)
    solutions = np.vstack(solutions_list)

    # Comparing with JAVA Implementation
    # path = os.path.join('C:\\Users\\luft7\\IdeaProjects\\IREOS-java\\solutions')
    # conv = {0: lambda x: 0 if x == b'"inlier"' else 1}
    # solutions = np.vstack([np.loadtxt(os.path.join(path, str(i)), converters=conv) for i in range(11)])

    ireos_setting = [
        (LogisticRegression, {'metric': 'probability', 'n_run_values': 100}),
        # (KLR, {'metric': 'probability', 'n_run_values': 100}),
        (LogisticRegression, {'metric': 'distance', 'n_run_values': 100}),
        (SVC, {'metric': 'probability', 'n_run_values': 100}),
        (SVC, {'metric': 'distance', 'n_run_values': 100}),
        (KNNM, {'n_run_values': 1}),
        (KNNM, {'n_run_values': int(0.1 * len(X))}),
        (KNNM, {'n_run_values': int(0.5 * len(X))}),
        (KNNC, {'n_run_values': 1}),
        (KNNC, {'n_run_values': int(0.1 * len(X))}),
        (KNNC, {'n_run_values': int(0.5 * len(X))}),
        (LinearSVC, {'metric': 'probability', 'n_run_values': 1}),
        (LinearSVC, {'metric': 'distance', 'n_run_values': 1}),
    ]

    C = 100

    for Maximum_Margin_Classifier, kwargs in ireos_setting:
        print(f'Compute correlation for {Maximum_Margin_Classifier.__name__} with {kwargs}')

        name = f'{Maximum_Margin_Classifier.__name__}_'
        if Maximum_Margin_Classifier in (SVC, LogisticRegression, LinearSVC):
            name += kwargs["metric"]
        elif Maximum_Margin_Classifier in (KNNC, KNNM):
            name += str(kwargs["n_run_values"])
        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0)
        except FileNotFoundError:
            results = pd.DataFrame([])
        if force_compute or name not in results.index:
            correlation = compute_spearman(Maximum_Margin_Classifier, X, y, solutions,
                                           C=C, m_cl=1, **kwargs)
            results.at[name, 'correlation'] = round(correlation, 3)
            results.to_csv(path)
        else:
            correlation = results.at[name, 'correlation']

        print(f'Correlation: {correlation}')


if __name__ == '__main__':
    main()
    exit(0)
