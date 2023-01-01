import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from data import load_campos_data, get_synthetic_features
from environment import Environment
from ireos import IREOS
from separability_algorithms import KLR, KNNC, KNNM, KNNC_w
from visualization import plot_classification

"""
This module visualizes the outlierness for chosen separability algorithms
on a certain dataset.
Dataset must be placed at ./datasets/prefix/!
input: datasetname.arff 
"""


def main():
    # X, y = get_synthetic_features(2, 3, 71, 0.015)
    dataset = "Parkinson_withoutdupl_norm_05_v04"
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    X, y = load_campos_data(dataset)
    pca = PCA(n_components=2)
    X_pca = np.copy(X)
    pca.fit_transform(X_pca)
    N = len(X)
    #plot_classification(X_pca, y)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=['orange' if e == 1 else 'blue' for e in y])
    plt.show()

    env: Environment = Environment()
    env.separability_algorithms = dict(n_samples=len(X_pca))

    first_case = True
    for name, classifier, kwargs in env.separability_algorithms:
        fig, ax = plt.subplots(figsize=(10, 10))

        application = IREOS(classifier, adjustment=True, **kwargs)
        application.fit(X, [y])
        c = list(application.compute_ireos_scores())
        outlierness = np.average(application.probability_array, axis=0)
        expected_value = np.average(outlierness)
        outlierness = (outlierness - expected_value) / (1 - expected_value)
        # print(np.average(application.probability_array, axis=1))
        cs = ax.scatter(X_pca[:, 0][y == 0], X_pca[:, 1][y == 0], c=outlierness[y == 0], cmap='winter')
        cs = ax.scatter(X_pca[:, 0][y == 1], X_pca[:, 1][y == 1], c=outlierness[y == 1], cmap='winter', marker='s')
        ax.set_title(f'{name}')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(cs)
        # cbar.ax.set_ylabel('outlierness')
        # cbar.ax.set_yticklabels([])
        # cbar.ax.set_yticks([])
        ax.annotate('Ground truth outliers' if first_case else 'Diverging outlierness labeling',
                    xy=(0.275, 0.09),
                    xycoords='data',
                    xytext=(0.8, 0.05), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8.0),
                    horizontalalignment='right', verticalalignment='top',
                    )
        first_case = False
        # fig.colorbar(cs)

        plt.show()

    exit(0)


if __name__ == '__main__':
    main()
