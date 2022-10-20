import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA


def plot_model(model, plot):
    h = .1
    xx, yy = np.meshgrid(np.arange(-8, 15, h),
                         np.arange(-8, 15, h))
    samples = np.concatenate((xx[:, :, None], yy[:, :, None]), axis=-1)
    samples = samples.reshape((np.prod(samples.shape[:-1]), samples.shape[-1]))

    Z = model.predict(samples).reshape((len(yy), len(xx)))
    Z = model.predict_proba(samples).reshape((len(yy), len(xx), 2))[:, :, 0]

    plot.contourf(xx, yy, Z, cmap='gray', vmin=0., vmax=1.)
    plot.axis('equal')


def plot_classification(X, y, plot=None, model=None, colors=None):
    if plot is None:
        fig, plot = plt.subplots()
    if model is not None:
        plot_model(model, plot)
    if X.shape[1] > 2:
        #TODO: An sich überlegen, ob jeden outlier in eine spezielle Klasse zu stecken Sinn macht
        lda = LinearDiscriminantAnalysis()
        #pca = PCA()
        nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
        pca = PCA(n_components=2, random_state=42)
        X = pca.fit_transform(X, (y >= 0.5).astype(int))
    if colors is None:
        colors = np.array(["blue", "orange"])
        plot.scatter(X[:, 0], X[:, 1], c=colors[(y >= 0.5).astype(int)])
    else:
        plot.scatter(X[:, 0], X[:, 1], c=colors, cmap='autumn')

    #ax[0].scatter(X[y == 0, 0], X[y == 0, 1], c=colors[y == 0])
    #ax[0].scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()
