import itertools
import os
import sys

from joblib import delayed, Parallel
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.utils._testing import ignore_warnings

from ireos import IREOS

sys.path.append('../../')

from matplotlib import pyplot as plt
import multiprocessing as mp
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from data import get_synthetic_features, create_data
from separability_algorithms import KLR, KNNM, KNNC
from scipy.integrate import simpson

from visualization import plot_model, plot_classification
import matplotlib.font_manager



debug = True
# chooseable is SVM, KLR or KNN
separability_algorithm = 'TEST'
used_kernel = 'rbf'
# area
measure = 'probability'
n_threads = mp.cpu_count()
grey_level = False
cmap = matplotlib.cm.get_cmap('viridis')
#colors = ['#e1165a', 'plum', 'lightblue']
colors = ['#e1165a', 'lightblue', 'plum']


def log(message):
    if debug:
        print(message)


def model_svm_rbf(X, Y, gamma, C):
    # works better with gamma between 0/1
    model = SVC(kernel='rbf', C=100, gamma=gamma, probability=False, random_state=0)

    Y_copy = np.copy(Y)
    if np.logical_or(Y == 0, Y == 1).all():
        Y_copy[Y == 0] = -1
    model.fit(X, Y_copy)
    if measure == 'probability':
        clf = CalibratedClassifierCV(model, cv="prefit", method='sigmoid')
        clf.fit(X, Y_copy)
        return clf
    elif measure == 'distance':
        return model
    # return clf


def model_klr(X, Y, gamma, C):
    # IDEA: gamma*gamma
    model = KLR(C=C, gamma=gamma)
    model.fit(X, Y)
    #model = KernelRidge(kernel='rbf', alpha=0.05, gamma=gamma)
    #model.fit(X, Y)
    #model = LogisticRegression(**dict(C=100, penalty='l1', intercept_scaling=0.5, solver='liblinear'))
    #f_map = Nystroem(gamma=gamma, random_state=0, n_components=len(X))
    #X_transformed = f_map.fit_transform(X)
    #model.fit(X_transformed, Y)
    return model


def model_log_regression(X, Y, gamma, C, *args):
    more_samples = np.repeat(X[Y == 1], len(Y == 0) - 2, axis=0)
    X_balanced = np.vstack((more_samples, X))
    Y_balanced = np.hstack((np.ones(len(more_samples)), Y))
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_balanced, Y_balanced)
    return model


def model_svm_linear(X, Y, gamma, C, *args):
    # more_samples = np.repeat(X[Y == 1], len(Y == 0) - 2, axis=0)
    # X_balanced = np.vstack((more_samples, X))
    # Y_balanced = np.hstack((np.ones(len(more_samples)), Y))
    model = LinearSVC(max_iter=10000, C=C)
    # model.fit(X_balanced, Y_balanced)
    model.fit(X, Y)
    return model


def model_log_linear(X, Y, gamma, C, *args):
    # more_samples = np.repeat(X[Y == 1], len(Y == 0) - 2, axis=0)
    # X_balanced = np.vstack((more_samples, X))
    # Y_balanced = np.hstack((np.ones(len(more_samples)), Y))
    model = LogisticRegression(max_iter=10000, C=C)
    # model.fit(X_balanced, Y_balanced)
    model.fit(X, Y)
    return model


def model_test(X, Y, k):
    Y_copy = np.copy(Y)
    # if np.logical_or(Y == 0, Y == 1).all():
    #    Y_copy[Y == 0] = -1
    more_samples = np.repeat(X[Y == 1], len(Y == 0) - 2, axis=0)
    X_balanced = np.vstack((more_samples, X))
    Y_balanced = np.hstack((np.ones(len(more_samples)), Y))
    # model = MLPClassifier(random_state=1, max_iter=k).fit(X_balanced, Y_balanced)
    # model = MLPClassifier(random_state=1, hidden_layer_sizes=(k,), max_iter=50*k).fit(X_balanced, Y_balanced)
    model = MLPClassifier(random_state=0, max_iter=k, solver='lbfgs').fit(X_balanced, Y_balanced)
    # print(model.n_iter_)
    return model


def model_knn(X, k):
    model = KNNC(k)
    model.fit(X)
    return model


def model_knn_m(X, k):
    model = KNNM(k)
    model.fit(X)
    return model
    # k = procent * (len(X) - 1)
    # model = KNNM(X, Y)
    # return model


model_functions = {
    'KLR': model_klr,
    'KLR_L': model_log_linear,
    'SVM': model_svm_rbf,
    'SVM_L': model_svm_linear,
    'LRG': model_log_regression,
    'KNN': model_knn,
    'KNN_M': model_knn_m,
    'TEST': model_test,
}


@ignore_warnings(category=ConvergenceWarning)
def main():



    # get data
    # X, y = get_synthetic_features(2, 3, 71, 0.015)
    # X, y = create_data(n_inlier=51, n_outlier=5)
    # X = preprocessing.StandardScaler().fit(X).transform(X)
    # X = preprocessing.MinMaxScaler().fit(X).transform(X)
    path = os.path.join('memory', 'example_set', 'example_1.npy')
    # if not os.path.exists(os.path.dirname(path)):
    #    os.makedirs(os.path.dirname(path))
    # np.save(path, X)
    X = np.load(path)
    # y = np.zeros(X.shape[0])
    y_1 = np.zeros(X.shape[0])
    y_1[0] = 1
    y_2 = np.zeros(X.shape[0])
    y_2[-1] = 1
    y_3 = np.zeros(X.shape[0])
    # 15 ganz gut, 31 auch, aber 30 ist das wahre
    y_3[30] = 1
    y_4 = np.zeros(X.shape[0])
    y_4[48] = 1
    y_c = y_1 + y_2 + y_3
    #plot_classification(X, y_4)
    #plt.show()



    # X += 100
    '''
    X = np.array([
#        [1, 1],
        [-1, -1],
#        [-1, 1],
        [1, -1],
        [0, 0],
        [0, 1],
    ])
    y = np.array([
#        0,
        0,
#        0,
        0,
        1,
        0])
    '''
    # oversampling
    # X[y == 1]

    # define gammas to evaluate
    n_gammas = 100
    max_gamma = 100

    gamma_values = np.linspace(0.0001, max_gamma, n_gammas)
    k_values = list(np.arange(1, 0.5 * len(X)).astype(int))
    #gamma_plot_values = np.array([0.2, 10, 100])
    gamma_plot_values = np.array([0.1, 1, 10, 100])
    k_plot_values = (np.array([0.01, 0.05, 0.15, 0.50]) * len(X)).astype(int)

    test_values = np.arange(1, 75)
    test_plot_values = (np.array([1, 10, 70])).astype(int)
    # gamma_values = np.logspace(-2, np.log10(max_gamma), 9)
    # for KNN{} use values between 0 and 1 as procent
    # gamma_values = np.logspace(-2, np.log10(max_gamma), n_gammas)

    # set arguments
    C = 100.0
    # C = np.array([100.0, 100.0, 100.0, 100.0])

    axes = []
    for i in range(4):
        axes.append(plt.subplots(figsize=(10, 10))[1])
        for direction in ['bottom', 'left', 'right', 'top']:
            axes[-1].spines[direction].set_linewidth(0.5)
            axes[-1].spines[direction].set_color('black')

    if grey_level:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

    # fig.suptitle(separability_algorithm)
    # fig.set_size_inches(18.5, 10.5)
    # fig.set_size_inches(25.5, 15.5)

    fig2, axis2 = plt.subplots()
    axis2.set_xticklabels([])
    axis2.set_xticks([])
    axis2.tick_params(axis=u'both', which=u'both', length=0)
    #plt.yticks(fontsize=12)
    plt.rcParams.update({'font.size': 12})
    axis2.spines["top"].set_visible(False)
    axis2.spines["right"].set_visible(False)
    axis2.spines["bottom"].set_linewidth(2)
    axis2.spines["bottom"].set_capstyle("round")
    axis2.spines["bottom"].set_color('black')
    #axis2.axis["xzero"].set_axisline_style("-|>")
    axis2.spines["left"].set_linewidth(2)
    axis2.spines["left"].set_capstyle("round")
    axis2.spines["left"].set_color('black')

    #axis2.spines["left"].set_visible(False)
    # axis2.set_xlim(0, gamma_values[-1])

    p_curves = []

    ireos_setting = [
        #('LRG_nystroem_p', LogisticRegression, dict(metric='probability', r_min=0.0001, r_max=max_gamma, sample_size=100, kernel_leverage='Nystroem',
        #                                            c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
        #                                                        solver='liblinear'))),
        ('LRG_nystroem_f', LogisticRegression, dict(metric='decision', r_min=0.0001, r_max=max_gamma, sample_size=100, kernel_leverage='Nystroem',
                                                    c_args=dict(C=100, penalty='l1', intercept_scaling=0.5,
                                                                solver='liblinear'))),
    ]

    if separability_algorithm == 'NLRG':
        for name, Maximum_Margin_Classifier, kwargs in ireos_setting:
            print(f'Compute correlation for {Maximum_Margin_Classifier.__name__} with {kwargs}')

            Ireos = IREOS(Maximum_Margin_Classifier, **kwargs)
            Ireos.fit(X, [y_c])
            c = list(Ireos.compute_ireos_scores())
            output_per_gamma = Ireos.probability_array
            p_curves = output_per_gamma.T[y_c.astype(bool)][[0, 2, 1]]

    for y in [y_1, y_3, y_2]:
    #for y in [y_4]:


    # plot_classification(X, y)
        arguments = map(lambda gamma: [X, y, gamma, C], list(gamma_values))
        if separability_algorithm.startswith('KNN'):
            print('here')
            arguments = map(lambda k: (X, k), k_values)
        elif separability_algorithm == 'TEST':
            arguments = map(lambda k: [X, y, k], test_values)
        # starting learning phase
        models = []
        from multiprocessing import Pool
        print(f'Starting with {n_threads} jobs')
        for model_indicator in separability_algorithm.split():
            log(f'Learning with algorithm: {model_indicator}...')
            trained_models = [delayed(model_functions[model_indicator])(*args) for args in arguments]
            models.append(Parallel(n_jobs=8)(trained_models))
            log(f'Training succeeded')



        # evaluate p-curves
        p = []
        for model in models:
            p_model = []
            outlier = X[y == 1]
            for model_variant in model:
                if separability_algorithm == 'TEST':
                    # p_model.append(model_variant.predict_proba(outlier)[0, 1])
                    p_model.append(model_variant.predict_proba(outlier)[0, 1])
                elif measure == 'probability':
                    p_model.append(model_variant.predict_proba(outlier)[0, 1])
                elif measure == 'distance':
                    p_model.append(model_variant.decision_function(outlier)[0])
                elif measure == 'predict':
                    p_model.append(model_variant.predict(outlier)[0])
                else:
                    raise NotImplementedError(f'{measure} not implemented')
            p.append(p_model)

        p = np.array(p)

        p_curves.append(p)

        # plotting model results
        for title, model in zip(separability_algorithm.split(), models):

            # gamma_plot_values = np.logspace(-3, np.log10(max_gamma), 4)
            # gamma_plot_values = np.linspace(0.0001, max_gamma, 4)
            if separability_algorithm.startswith('KNN'):
                models = [model_functions[title](X, index) for index in k_plot_values]
            elif separability_algorithm == 'TEST':
                models = [model_functions[title](X, y, index) for index in test_plot_values]
            else:
                models = [model_functions[title](X, y, index, C) for index in gamma_plot_values]
            # axes = np.array(axis).reshape(-1)
            cs = plot_model(models, axes, proba=measure, grey_level=grey_level, outlier=X[y == 1])
            if grey_level:
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbarx = fig.colorbar(cs, cax=cbar_ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], )
                cbarx.ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=16)

            # for index, ax in zip(gamma_plot_values, np.array(axis).reshape(-1)):
            # model_variant = model[index]
            # model_variant = model_functions[title](X, y, index, C)
            # gamma = gamma_values[index]
            # plot_model(model_variant, ax)
            # ax.set_title(f"gamma = {gamma:.5f}")
            # ax.set_title(f"p = {model_variant.predict_proba(outlier)[0, 1]:.5f}")
            # print(model_variant.predict_proba(outlier)[0, 1])
            # print(model_variant.predict_proba(outlier)[0, 1])
            # ax.scatter(X[y == 0, 0], X[y == 0, 1])
            # ax.scatter(X[y == 1, 0], X[y == 1, 1])

    # Plot gamma curve
    # p_curves = np.array(p_curves)
    # p_curves = (p_curves - p_curves.min()) / (p_curves.max() - p_curves.min())

    if separability_algorithm == 'TEST':
        axis2.set_xlabel("I", fontsize=18, fontname='Calibri')
        axis2.set_ylabel("p", fontsize=18, fontname='Calibri')
        axis2.set_ylim(0, 1.1)
    elif separability_algorithm.startswith('KNN'):
        axis2.set_xlabel("k", fontsize=18, fontname='Calibri')
        axis2.set_ylabel("d", fontsize=18, fontname='Calibri')
        axis2.set_yticks(np.array([0, 0.5, 1]), )
        axis2.set_ylim(0, 1.1)
    elif measure == 'probability':
        axis2.set_xlabel("\u03B3", fontsize=18, fontname='Calibri')
        axis2.set_ylabel("p", fontsize=18, fontname='Calibri')
        axis2.set_ylim(0, 1.1)
    else:
        axis2.set_ylim(np.floor(np.min(p_curves)), np.ceil(np.max(p_curves)))
        axis2.set_xlabel("\u03B3", fontsize=18, fontname='Calibri')
        axis2.set_ylabel("f", fontsize=18, fontname='Calibri')

    for index, p_curve in enumerate(p_curves):
        if separability_algorithm != 'NLRG':
            print(np.sum(p_curve) / len(p_curve[0]))
        else:
            print(f'{np.average(p_curve): .2f}')
        axis2.plot((1), (np.floor(np.min(p_curves))), ls="", marker=">", ms=10, color="black",
                   transform=axis2.get_yaxis_transform(), clip_on=False, zorder=100)
        if separability_algorithm.startswith('KNN'):
            axis2.plot(k_values, p_curve[0], color=colors[index])
        elif separability_algorithm == 'TEST':
            axis2.plot(test_values, p_curve[0], color=colors[index])
        elif separability_algorithm == 'NLRG':
            axis2.plot(gamma_values, p_curve, color=colors[index])
        else:
            axis2.plot(gamma_values, p_curve[0], color=colors[index])
        # axis2.legend(loc='lower right')

    if not grey_level:
        for ax in axes:
            ax.scatter(X[y_c == 0, 0], X[y_c == 0, 1], zorder=1, color='black')
            ax.scatter(X[y_c == 1, 0], X[y_c == 1, 1], zorder=1, color=colors, marker='s')
    else:
        for ax in axes:
            ax.scatter(X[y_4 == 0, 0], X[y_4 == 0, 1], zorder=1, color='blue')
            ax.scatter(X[y_4 == 1, 0], X[y_4 == 1, 1], zorder=1, color='orange')

    #plt.style.use('seaborn-whitegrid')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)
