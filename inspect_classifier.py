import itertools
import os
import sys

from joblib import delayed, Parallel
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

sys.path.append('../../')

from matplotlib import pyplot as plt
import multiprocessing as mp
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data import get_synthetic_features, create_data
from log_regression import KLR, KNNM, KNNC, KNNC_inspect, KNNM_inspect

from visualization import plot_model, plot_classification

debug = True
# chooseable is SVM, KLR or KNN
separability_algorithm = 'TEST'
used_kernel = 'rbf'
# area
measure = 'probability'
n_threads = mp.cpu_count()


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
    #return clf


def model_klr(X, Y, gamma, C):
    # IDEA: gamma*gamma
    model = KLR(C=C, gamma=gamma)
    model.fit(X, Y)
    return model


def model_log_regression(X, Y, gamma):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)
    return model

def model_test(X, Y, k):
    Y_copy = np.copy(Y)
    if np.logical_or(Y == 0, Y == 1).all():
        Y_copy[Y == 0] = -1
    model = MLPClassifier(random_state=0, max_iter=3000, hidden_layer_sizes=(100, k + 1)).fit(X, Y_copy)
    return model



def model_knn(X, k):
    return KNNC_inspect(X, k)


def model_knn_m(X, k):
    return KNNM_inspect(X, k)
    #k = procent * (len(X) - 1)
    #model = KNNM(X, Y)
    #return model


model_functions = {
    'KLR': model_klr,
    'SVM': model_svm_rbf,
    'LRG': model_log_regression,
    'KNN': model_knn,
    'KNN_M': model_knn_m,
    'TEST': model_test,
}


def main():
    # get data
    #X, y = get_synthetic_features(2, 3, 71, 0.015)
    # X, y = create_data(n_inlier=51, n_outlier=5)
    #X = preprocessing.StandardScaler().fit(X).transform(X)
    #X = preprocessing.MinMaxScaler().fit(X).transform(X)
    path = os.path.join('memory', 'example_set', 'example_1.npy')
    #if not os.path.exists(os.path.dirname(path)):
    #    os.makedirs(os.path.dirname(path))
    #np.save(path, X)
    X = np.load(path)
    #y = np.zeros(X.shape[0])
    y_1 = np.zeros(X.shape[0])
    y_1[0] = 1
    y_2 = np.zeros(X.shape[0])
    y_2[-1] = 1
    y_3 = np.zeros(X.shape[0])
    # 15 ganz gut, 31 auch
    y_3[30] = 1
    y_c = y_1 + y_2 + y_3

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
    k_values = list(np.arange(1, 0.5*len(X)).astype(int))
    gamma_plot_values = np.array([0.10, 1, 10, 100])
    k_plot_values = (np.array([0.01, 0.05, 0.15, 0.50])*len(X)).astype(int)
    # gamma_values = np.logspace(-2, np.log10(max_gamma), 9)
    # for KNN{} use values between 0 and 1 as procent
    # gamma_values = np.logspace(-2, np.log10(max_gamma), n_gammas)

    # set arguments
    C = 100.0
    # C = np.array([100.0, 100.0, 100.0, 100.0])

    fig, axis = plt.subplots(2, 2)
    fig.suptitle(separability_algorithm)
    fig.set_size_inches(18.5, 10.5)

    fig2, axis2 = plt.subplots()
    if measure == 'probability' or separability_algorithm.startswith('KNN'):
        axis2.set_ylim(0, 1.1)
    elif separability_algorithm == 'TEST':
        axis2.set_ylim(0, 1.1)
    else:
        axis2.set_ylim(-1.1, 1.1)
    #axis2.set_xlim(0, gamma_values[-1])

    p_curves = []

    for y in [y_1, y_2, y_3]:

        #plot_classification(X, y)
        arguments = map(lambda gamma: [X, y, gamma, C], list(gamma_values))
        if separability_algorithm.startswith('KNN'):
            arguments = map(lambda k: [X, k], k_values)
        elif separability_algorithm == 'TEST':
            arguments = map(lambda k: [X, y, k], k_values)
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
                if measure == 'probability':
                    p_model.append(model_variant.predict_proba(outlier)[0, 1])
                elif measure == 'distance':
                    p_model.append(model_variant.decision_function(outlier)[0])
                else:
                    raise NotImplementedError(f'{measure} not implemented')
            p.append(p_model)

        p = np.array(p)

        p_curves.append(p)

        # plotting model results
        for title, model in zip(separability_algorithm.split(), models):

            #gamma_plot_values = np.logspace(-3, np.log10(max_gamma), 4)
            #gamma_plot_values = np.linspace(0.0001, max_gamma, 4)
            if separability_algorithm.startswith('KNN'):
                models = [model_functions[title](X, index) for index in k_plot_values]
            elif separability_algorithm == 'TEST':
                models = [model_functions[title](X, y, index) for index in k_plot_values]
            else:
                models = [model_functions[title](X, y, index, C) for index in gamma_plot_values]
            axes = np.array(axis).reshape(-1)
            plot_model(models, axes, proba=measure=='probability')

            #for index, ax in zip(gamma_plot_values, np.array(axis).reshape(-1)):
                # model_variant = model[index]
                #model_variant = model_functions[title](X, y, index, C)
                # gamma = gamma_values[index]
                #plot_model(model_variant, ax)
                # ax.set_title(f"gamma = {gamma:.5f}")
                #ax.set_title(f"p = {model_variant.predict_proba(outlier)[0, 1]:.5f}")
                #print(model_variant.predict_proba(outlier)[0, 1])
                #print(model_variant.predict_proba(outlier)[0, 1])
                #ax.scatter(X[y == 0, 0], X[y == 0, 1])
                #ax.scatter(X[y == 1, 0], X[y == 1, 1])

    # Plot gamma curve
    #p_curves = np.array(p_curves)
    #p_curves = (p_curves - p_curves.min()) / (p_curves.max() - p_curves.min())

    for p_curve in p_curves:
        #print(p_model)
        if separability_algorithm.startswith('KNN'):
            axis2.plot(k_values, p_curve[0])
        elif separability_algorithm == 'TEST':
            axis2.plot(k_values, p_curve[0])
        else:
            axis2.plot(gamma_values, p_curve[0])
        # axis2.legend(loc='lower right')


    for ax in np.array(axis).reshape(-1):
        ax.scatter(X[y_c == 0, 0], X[y_c == 0, 1], zorder=1)
        ax.scatter(X[y_c == 1, 0], X[y_c == 1, 1], zorder=1)

    plt.show()


if __name__ == '__main__':
    main()
    exit(0)