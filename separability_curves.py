import os

import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
import numpy.typing as npt
from ireos import IREOS
from visualization import plot_classification



"""
This module visualizes the separability curves for a given separability algorithm
on a certain dataset.
Dataset must be placed at ./datasets/Example/!
input: datasetname.npy 
"""



def main():
    path = os.path.join('datasets', 'Example', 'example_1.npy')
    X: npt.ArrayLike = np.load(path)
    n_samples = len(X)
    n_dimensions = len(X[0])

    env: Environment = Environment()
    env.separability_algorithms = dict(n_samples=n_samples)

    samples_of_interest = [0, 30, -1]
    y = np.zeros(len(X))
    y[samples_of_interest] = 1
    plot_classification(X, y)
    plt.show()

    colors = ['#e1165a', 'lightblue', 'plum']
    # axes = []
    # for i in range(4):
    #    axes.append(plt.subplots(figsize=(10, 10))[1])
    #    for direction in ['bottom', 'left', 'right', 'top']:
    #        axes[-1].spines[direction].set_linewidth(0.5)
    #        axes[-1].spines[direction].set_color('black')

    for name, Maximum_Margin_Classifier, kwargs in env.separability_algorithms:
        print(f'Compute probability array for {Maximum_Margin_Classifier.__name__} with {kwargs}')
        application: IREOS = IREOS(Maximum_Margin_Classifier, **kwargs)
        application.fit(X, [y])
        index_scores = list(application.compute_ireos_scores())
        separability_curves = application.probability_array[:, samples_of_interest].T

        curves_fig, curves_ax = plt.subplots()
        #curves_ax.set_xticklabels([])
        #curves_ax.set_xticks([])
        curves_ax.tick_params(axis=u'both', which=u'both', length=0)
        # plt.yticks(fontsize=12)
        plt.rcParams.update({'font.size': 12})
        curves_ax.spines["top"].set_visible(False)
        curves_ax.spines["right"].set_visible(False)
        curves_ax.spines["bottom"].set_linewidth(2)
        curves_ax.spines["bottom"].set_capstyle("round")
        curves_ax.spines["bottom"].set_color('black')
        # axis2.axis["xzero"].set_axisline_style("-|>")
        curves_ax.spines["left"].set_linewidth(2)
        curves_ax.spines["left"].set_capstyle("round")
        curves_ax.spines["left"].set_color('black')
        curves_ax.plot((1), (np.floor(np.min(application.probability_array))), ls="", marker=">", ms=10, color="black",
                       transform=curves_ax.get_yaxis_transform(), clip_on=False, zorder=100)
        curves_ax.set_ylim(np.floor(np.min(separability_curves)), np.ceil(np.max(separability_curves)))

        for index, p_curve in enumerate(separability_curves):
            print(f'{np.average(p_curve): .2f}')
            curves_ax.plot(application.r_values, p_curve)

        plt.show()

        # if separability_algorithm.startswith('KNN'):
        #    axis2.plot(k_values, p_curve[0], color=colors[index])
        # elif separability_algorithm == 'TEST':
        #    axis2.plot(test_values, p_curve[0], color=colors[index])
        # elif separability_algorithm == 'NLRG':
        #    axis2.plot(gamma_values, p_curve, color=colors[index])
        # else:
        #    axis2.plot(gamma_values, p_curve[0], color=colors[index])

        # for ax in axes:
        #    ax.scatter(X[y == 0, 0], X[y == 0, 1], zorder=1, color='black')
        #    ax.scatter(X[y == 1, 0], X[y == 1, 1], zorder=1, color=colors, marker='s')

    pass


if __name__ == '__main__':
    main()
    exit(0)
