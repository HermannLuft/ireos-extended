import os
import sys

import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PCA_Detector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from main import compute_outlier_result

from ireos import IREOS
from data import load_campos_data

import numpy as np
from separability_algorithms import KLR, KNNC, KNNM

compute_ireos_difficulty = False

"""
This module computes the difficulty, diversity or hardness for a dataset
Dataset must be placed at ./datasets/prefix/!
input: datasetname.arff 
"""

def main():
    """
    Computes difficulty and diversity for dataset according to
    Outlier Detection results. Takes dataset as argument
    """

    dataset = "Hepatitis_withoutdupl_norm_05_v02"
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)

    N = len(X)
    d = len(X[0])

    # Algorithms to compute the difficutly and diversity
    algorithm_setting = [
        dict(ad_algorithm=ABOD, r_name='n_neighbors', interval=(1, min(100, N))),
        dict(ad_algorithm=KDE, r_name='bandwidth', sampling_size=100),
        dict(ad_algorithm=COF, r_name='n_neighbors', interval=(2, min(100, N))),
        dict(ad_algorithm=LOF, r_name='n_neighbors', interval=(1, min(100, N))),
        dict(ad_algorithm=KNN, r_name='n_neighbors', interval=(1, min(100, N - 1))),
        dict(ad_algorithm=IForest, r_name='n_estimators', interval=(1, min(100, N))),
        dict(ad_algorithm=OCSVM, r_name='nu', interval=(0.25, 0.75), sampling_size=100),
        dict(ad_algorithm=PCA_Detector, r_name='n_components', interval=(1, min(N, d))),
        # dict(ad_algorithm=LODA, r_name='n_random_cuts', interval=(1, min(100, N)),
        #     ad_kwargs={'n_bins': 'auto'}),
        dict(ad_algorithm=HBOS, r_name='n_histograms', interval=(3, min(100, N))),
    ]

    classifier_setting = [
        ('KLR_p', KLR, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100,))),
        ('KLR_f', KLR, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100,))),
        ('LRG_nystroem_p', LogisticRegression, dict(metric='probability', sample_size=100, kernel_leverage='Nystroem',
                                  c_args=dict(C=100, penalty='l1', intercept_scaling=0.5, solver='liblinear'))),
        ('LRG_nystroem_f', LogisticRegression, dict(metric='decision', sample_size=100, kernel_leverage='Nystroem',
                                  c_args=dict(C=100, penalty='l1', intercept_scaling=0.5, solver='liblinear'))),
        ('SVM_p', SVC, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100,
                                                                     probability=False, random_state=0))),
        ('SVM_f', SVC, dict(r_name='gamma', metric='decision', sample_size=100, c_args=dict(kernel='rbf', C=100,
                                                                  probability=False, random_state=0))),
        ('LRG_linear', LogisticRegression, dict(metric='probability', c_args=dict(C=100))),
        ('SVM_linear', LinearSVC, dict(metric='decision', c_args=dict(C=100))),
        ('KNNC_10%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * N))),
        ('KNNC_50%', KNNC, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * N))),
        ('KNNM_10%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.1 * N))),
        ('KNNM_50%', KNNM, dict(r_name='k', metric='decision', r_min=1, r_max=int(0.5 * N))),
        ('MLP', MLPClassifier, dict(r_name='max_iter', metric='probability', balance_class=True,
                             r_min=1, r_max=25,
                             c_args=dict(random_state=0))),
    ]

    n = len(y[y == 1])
    ranked_map = np.empty((n, len(algorithm_setting)))

    for alg_index, kwargs in enumerate(algorithm_setting):
        # Get best auc score solution for different k's
        solutions = compute_outlier_result(X, dataset, **kwargs)
        roc_auc_evaluations = [roc_auc_score(y, solution) for solution in solutions]
        argmax, _ = max(list(enumerate(roc_auc_evaluations)), key=lambda x: x[1])
        best_solution = solutions[argmax]

        # get outlier ranks according to top n
        outlier_indices, = np.where(y == 1)
        _, positions = np.where(np.argsort(best_solution)[::-1] == outlier_indices[:, None])
        bins = positions // n + 1
        ranked_map[:, alg_index] = np.clip(bins, None, 10)

    path = os.path.join('memory', dataset, 'dataset_evaluation.csv')
    results = pd.DataFrame([])

    difficulty = np.average(ranked_map)
    x = np.std(ranked_map, axis=1)
    diversity = np.sqrt(x.dot(x) / x.size)

    print(f'Difficulty: {difficulty}')
    print(f'Diversity: {diversity}')

    if compute_ireos_difficulty:
        for name, Maximum_Margin_Classifier, kwargs in classifier_setting:
            Ireos = IREOS(Maximum_Margin_Classifier, adjustment=True, **kwargs)
            Ireos.fit(X, [y])
            c = list(Ireos.compute_ireos_scores())
            results.at[f'index_{name}', 'evaluation'] = c[0]
            print(f'Datasetscore: {c}')


    results.at['Difficulty', 'evaluation'] = difficulty
    results.at['Diversity', 'evaluation'] = diversity
    results.to_csv(path)




    exit(0)


if __name__ == '__main__':
    main()
    exit(0)
