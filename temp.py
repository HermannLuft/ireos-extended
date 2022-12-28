import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing

from data import get_synthetic_features
from visualization import confidence_ellipse


def main():
    fig, ax = plt.subplots()
    ax.plot()
    plt.show()
    exit(0)
    X, _ = get_synthetic_features(2, 3, 100, 0.5)
    X_p = preprocessing.MinMaxScaler((-0.5, 0.5)).fit(X).transform(X)
    X_1 = X_p[np.linalg.norm(X_p, axis=1) >= 0.2]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter(X_1[:, 0], X_1[:, 1])
    ax.scatter(0, 0, color='orange')


    fig, ax = plt.subplots(figsize=(6, 6))
    X_2, _ = get_synthetic_features(2, 0.01, 5, 0)
    X_2 = X_2 + 1
    X_c = np.concatenate((X, X_2))

    X_p = preprocessing.MinMaxScaler((-0.5, 0.5)).fit(X_c).transform(X_c)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter(X_p[:-5, 0], X_p[:-5, 1])
    ax.scatter(X_p[-5:, 0], X_p[-5:, 1], color='orange')



    # creating univariate mixed attiributes outlier

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter(X_1[:, 0]*0.99, np.zeros(X_1[:, 0].shape))
    ax.scatter(0.9, 0, color='orange', marker='s')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), gridspec_kw={'height_ratios': [4, 1]})
    for ax_entry in ax.flatten():
        ax_entry.set_yticklabels([])
        ax_entry.set_xticklabels([])
        ax_entry.set_xticks([])
        ax_entry.set_yticks([])

    mu_1, sigma_1 = 0, 0.2  # mean and standard deviation
    s1 = np.random.normal(mu_1, sigma_1, size=(100, 1))
    mu_2, sigma_2 = 1, 0.4  # mean and standard deviation
    s2 = np.random.normal(mu_2, sigma_2, size=(100, 1))
    X_3 = np.hstack((s1, s2))
    #X_p = preprocessing.MinMaxScaler((-0.5, 0.5)).fit(X_3).transform(X_3)


    print(X.shape)
    ax[0, 0].set_aspect('equal')
    ax[0, 0].set(xlim=(-1.8, 1.8), ylim=(-0.8, 2.8))
    ax[0, 0].scatter(X_3[:, 0], X_3[:, 1])
    ax[0, 0].scatter([1.5, 0.9, -1.3], [1.5, 0.9, -0.5], color='orange')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')

    t = np.linspace(0, 2 * np.pi, 100)
    ax[0, 0].plot(mu_1 + 4*sigma_1 * np.cos(t), mu_2 + 4*sigma_2 * np.sin(t), color='darkred')

    ax[1, 0].set(xlim=(-2.5, 2.5), ylim=(0, 2.0))
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('f')
    x = np.linspace(-2.5, 2.5, 300)
    ax[1, 0].plot(x, stats.norm.pdf(x, mu_1, sigma_1))


    ax[1, 1].set(xlim=(-2.5, 2.5), ylim=(0, 2.0))
    ax[1, 1].set_xlabel('y')
    ax[1, 1].set_ylabel('f')
    x = np.linspace(-2.5, 2.5, 300)
    ax[1, 1].plot(x, stats.norm.pdf(x, 0, sigma_2))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    X_2, _ = get_synthetic_features(2, 3, 30, 0.5)
    X_2 = X_2 + 2
    ax.scatter(X[:, 0], X[:, 1])
    ax.scatter(X_2[:, 0], X_2[:, 1], color='orange')

    # ax.scatter(X_2[:, 0], X_2[:, 1], color='orange')
    plt.show()



if __name__ == '__main__':
    main()
    exit(0)
