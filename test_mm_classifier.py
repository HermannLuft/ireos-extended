import itertools
import sys

from joblib import delayed, Parallel
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV

sys.path.append('../../')

from matplotlib import pyplot as plt
import multiprocessing as mp
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from noise.ireos.classifier.threshold import ThresholdClassifier
from noise.ireos.data import get_synthetic_features, create_data
from noise.ireos.log_regression import KLR, KNNM, KNNC

from noise.ireos.visualization import plot_model, plot_classification

debug = True
# chooseable is SVM, KLR or KNN
separability_algorithm = 'SVM'
used_kernel = 'rbf'
# area
measure = 'probability'
n_threads = mp.cpu_count()


def log(message):
    if debug:
        print(message)


def model_svm_rbf(X, Y, gamma, C):
    # works better with gamma between 0/1
    X_oversampled = np.vstack((X, np.repeat(X[Y == 1], len(Y[Y == 0]), axis=0)))
    Y_oversampled = np.hstack((Y, np.ones(len(Y[Y == 0]))))
    model = SVC(kernel='rbf', C=100, gamma=gamma, probability=False, random_state=0)
    Y_copy = np.copy(Y)
    if np.logical_or(Y == 0, Y == 1).all():
        Y_copy[Y == 0] = -1
    clf = CalibratedClassifierCV(model, cv=3)
    clf.fit(X, Y_copy)
    return clf


def model_klr(X, Y, gamma, C):
    # IDEA: gamma*gamma
    model = KLR(kernel=used_kernel, C=C, gamma=gamma)
    model.fit(X, Y)
    return model


def model_log_regression(X, Y, gamma):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)
    return model


def model_knn(X, Y, procent, C):
    k = procent * (len(X) - 1)
    model = KNNC(X, Y)
    return model


def model_knn_m(X, Y, procent, C):
    k = procent * (len(X) - 1)
    model = KNNM(X, Y)
    return model


model_functions = {
    'KLR': model_klr,
    'SVM': model_svm_rbf,
    'LRG': model_log_regression,
    'KNN': model_knn,
    'KNN_M': model_knn_m,
}


def main():
    # get data
    X, y = get_synthetic_features(2, 3, 71, 0.05)
    # X, y = create_data(n_inlier=51, n_outlier=5)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    plot_classification(X, y)
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
    max_gamma = 1

    gamma_values = np.linspace(0.0001, max_gamma, n_gammas)
    # gamma_values = np.logspace(-2, np.log10(max_gamma), 9)
    # for KNN{} use values between 0 and 1 as procent
    # gamma_values = np.logspace(-2, np.log10(max_gamma), n_gammas)

    # set arguments
    C = 100.0
    # C = np.array([100.0, 100.0, 100.0, 100.0])
    arguments = map(lambda gamma: [X, y, gamma, C], list(gamma_values))

    # starting learning phase
    models = []
    from multiprocessing import Pool
    print(f'Starting with {n_threads} jobs')
    for model_indicator in separability_algorithm.split():
        log(f'Learning with algorithm: {model_indicator}...')
        trained_models = [delayed(model_functions[model_indicator])(X, y, gamma, C)
                          for gamma in gamma_values]
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

    print(p)

    # plotting model results
    for title, model in zip(separability_algorithm.split(), models):
        fig, axis = plt.subplots(3, 3)
        fig.suptitle(title)
        fig.set_size_inches(18.5, 10.5)
        # gamma_plot_values = np.logspace(-2, np.log10(max_gamma), 9)
        gamma_plot_values = np.linspace(0.0001, max_gamma, 9)
        for index, ax in zip(gamma_plot_values, np.array(axis).reshape(-1)):
            # model_variant = model[index]
            model_variant = model_functions[title](X, y, index, C)
            # gamma = gamma_values[index]
            gamma = index
            plot_model(model_variant, ax)
            # ax.set_title(f"gamma = {gamma:.5f}")
            ax.set_title(f"p = {model_variant.predict_proba(outlier)[0, 1]:.5f}")
            print(model_variant.predict_proba(outlier)[0, 1])
            print(model_variant.predict_proba(outlier)[0, 1])
            ax.scatter(X[y == 0, 0], X[y == 0, 1])
            ax.scatter(X[y == 1, 0], X[y == 1, 1])

    # Plot gamma curve
    fig2, axis2 = plt.subplots()
    axis2.set_ylim(0, 1.1)
    axis2.set_xlim(0, gamma_values[-1])
    ret = None
    for p_model, label in zip(p, separability_algorithm.split()):
        print(p_model)
        ret = axis2.plot(gamma_values, p_model, label=label)
    # axis2.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
    exit(0)
