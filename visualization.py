import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA

Z_list = None
index = 0
cmaps = ['Purples', 'Oranges', 'Greens']

def plot_model(models, plots, proba=True):
    global Z_list, index
    h = 1000
    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, h),
                         np.linspace(-0.1, 1.1, h))

    for model, plot in zip(models, plots):
        samples = np.concatenate((xx[:, :, None], yy[:, :, None]), axis=-1)
        samples = samples.reshape((np.prod(samples.shape[:-1]), samples.shape[-1]))

        #Z = model.predict(samples).reshape((len(yy), len(xx)))
        if proba:
            Z = model.predict_proba(samples).reshape((len(yy), len(xx), 2))[:, :, 1]
        else:
            Z = model.decision_function(samples).reshape((len(yy), len(xx)))
        #Z = (Z - Z.min())/(Z.max() - Z.min())
        #print(Z)
        #Z[Z < 0.99] = 0


        if proba:
            levels = np.linspace(0, 1, 4)
            plot.contour(xx, yy, Z, cmap=cmaps[index], vmin=0, vmax=1, levels=levels, zorder=-1)
        else:
            levels = np.linspace(-1, 1, 4)
            plot.contour(xx, yy, Z, cmap=cmaps[index], vmin=-1, vmax=1, levels=levels, zorder=-1)
        plot.autoscale(False)
        #plot.set_aspect('equal', 'box')

    index += 1
    #plot.axis('equal', 'box')


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
    #plt.show()
