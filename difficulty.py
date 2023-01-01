import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from environment import Environment
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

    env: Environment = Environment()
    env.anomaly_detection = dict(n_samples=N, n_dimensions=d)
    env.separability_algorithms = dict(n_samples=N)
    n = len(y[y == 1])
    ranked_map = np.empty((n, len(env.anomaly_detection)))

    for alg_index, kwargs in enumerate(env.anomaly_detection):
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
        for name, Maximum_Margin_Classifier, kwargs in env.separability_algorithms:
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
