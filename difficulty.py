import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from outlier_detection import FastABOD, LDF, LOF, LoOP, COF, KNN, IsoForest, OC_SVM, PCA_Detector

sys.path.append('../../')
from ireos import IREOS, IREOS_LC
from data import load_campos_data

import numpy as np
from log_regression import KLR


compute_ireos_difficulty = True

# TODO: similiarity with main compute_outlier_resuls
def get_outlier_results(algorithm, X, y, dataset=None, **kwargs):
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


'''
Parkinson 4: das schlechteste
Parkinson 2: perfekt
'''


def main():
    dataset = "Hepatitis_withoutdupl_norm_05_v02"
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)

    N = len(X)

    algorithm_setting = [
        (FastABOD, {'max_neighbors': N}),
        (LDF, {'max_neighbors': N}),
        (LOF, {'max_neighbors': N}),
        (LoOP, {'max_neighbors': N}),
        (COF, {'max_neighbors': N}),
        (KNN, {'max_neighbors': N}),
        (IsoForest, {'max_neighbors': N}),
        (OC_SVM, {'max_neighbors': N}),
        # (LOCI, {'max_neighbors': N}), Not further used due to time reasons
        (PCA_Detector, {'max_neighbors': N}),
    ]

    n = len(y[y == 1])
    ranked_map = np.empty((n, len(algorithm_setting)))

    for alg_index, (algorithm, kwargs) in enumerate(algorithm_setting):
        solutions = get_outlier_results(algorithm, X, y, dataset, **kwargs)
        roc_auc_evaluations = [roc_auc_score(y, solution) for solution in solutions]
        argmax, _ = max(list(enumerate(roc_auc_evaluations)), key=lambda x: x[1])
        best_solution = solutions[argmax]
        outlier_indices, = np.where(y == 1)
        _, positions = np.where(np.argsort(best_solution)[::-1] == outlier_indices[:, None])
        bins = positions // n + 1
        ranked_map[:, alg_index] = np.clip(bins, None, 10)

    difficulty = np.average(ranked_map)
    x = np.std(ranked_map, axis=1)
    diversity = np.sqrt(x.dot(x) / x.size)

    print(f'Difficulty: {difficulty}')
    print(f'Diversity: {diversity}')

    if compute_ireos_difficulty:
        Ireos = IREOS_LC(LogisticRegression, adjustment=True)
        Ireos.fit(X, [y])
        c = list(Ireos.compute_ireos_scores())
        print(f'Datasetscore: {c}')

    # print(f'E_I: {Ireos.E_I}')

    path = os.path.join('memory', dataset, 'dataset_evaluation.csv')
    results = pd.DataFrame([])
    results.at['Difficulty', 'evaluation'] = difficulty
    results.at['Diversity', 'evaluation'] = diversity
    if compute_ireos_difficulty:
        results.at['IREOS_index', 'evaluation'] = c[0]
    results.to_csv(path)

    exit(0)


if __name__ == '__main__':
    main()
    exit(0)
