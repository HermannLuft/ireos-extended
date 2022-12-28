import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from scipy import interpolate
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC, LinearSVC
from matplotlib import ticker, cm

from separability_algorithms import KLR, KNNC

Z_list = None
index = 0
colors = ['#e1165a', 'plum', 'lightblue']

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Creates confidence elipse on plot ax around data points in dimensions x and y
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    if 'group' in kwargs.keys():
        group_txt = kwargs.pop('group')
        ax.annotate(group_txt, (mean_x + 0.05, mean_y + 0.05), fontsize=14)


    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)



    ax.scatter(mean_x, mean_y, marker='x', color=kwargs['edgecolor'])



    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def create_colormap(first_color, last_color):
    """
    Creating colormap with first_color reaching to 0.5
    and the rest is uniformly sampled to last_color
    """
    hex_to_rgb = lambda x: np.array([*[int(x[i:i + 2], 16) for i in (0, 2, 4)], 256]).astype(float) / 256
    # second_color =
    own_cmap = np.array([hex_to_rgb(first_color),
                         hex_to_rgb(last_color)])
    f = interpolate.interp1d(np.array([0, 1]), own_cmap, axis=0)
    own_cmap = f(np.linspace(0, 1, 256))
    own_cmap[:128] = hex_to_rgb(first_color)
    # own_cmap = np.interp(np.linspace(0, 1, 256), np.array([0, 1]), own_cmap)
    return ListedColormap(own_cmap)


def plot_model(models, plots, proba='probability', grey_level=False, outlier=None):
    """
    plots model predictions into plots as contours
    """
    global index
    cs = None
    h = 1000

    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, h),
                         np.linspace(-0.1, 1.1, h))

    for model, plot in zip(models, plots):
        samples = np.concatenate((xx[:, :, None], yy[:, :, None]), axis=-1)
        samples = samples.reshape((np.prod(samples.shape[:-1]), samples.shape[-1]))

        #Z = model.predict(samples).reshape((len(yy), len(xx)))
        if proba == 'probability':
            Z = model.predict_proba(samples).reshape((len(yy), len(xx), 2))[:, :, 1]
        elif proba == 'distance':
            Z = model.decision_function(samples).reshape((len(yy), len(xx)))
        elif proba == 'predict':
            Z = model.predict(samples).reshape((len(yy), len(xx)))
        else:
            raise NotImplementedError('metric not implemented!')
        #Z = (Z - Z.min())/(Z.max() - Z.min())
        #print(Z)
        #Z[Z < 0.99] = 0

        plot.set_yticklabels([])
        plot.set_xticklabels([])
        plot.set_xticks([])
        plot.set_yticks([])
        #if isinstance(model, KLR):
        #    plot.set_title(f"\u03B3-value: {model.get_gamma()} and p: {round(model.predict_proba(outlier)[0, 1], 2)}")
        #elif isinstance(model, SVC):
        #    plot.set_title(f"\u03B3-value: {model._gamma}")
        #elif isinstance(model, KNNC):
        #    plot.set_title(f"\u03B3-value: {model.get_k()}")
        if grey_level:
            plot.set_title(f"p: {round(model.predict_proba(outlier)[0, 1], 2)}", fontsize=18)
            cs = plot.contourf(xx, yy, Z, cmap='gist_gray_r', levels=np.linspace(0, 1, 21), vmin=0., vmax=1.0001,
                               orientation='horizontal')
        elif proba == 'probability':
            levels = np.linspace(0, 1, 4)
            plot.contour(xx, yy, Z, colors=colors[index], vmin=0, vmax=1, levels=[0.5], zorder=-1)
        else:
            levels = np.linspace(-1, 1, 1)
            absent = np.max([np.abs(np.min(Z)), np.abs(np.max(Z))])
            plot.contour(xx, yy, Z, colors=colors[index], vmin=-absent, vmax=absent, levels=[0], zorder=-1)
        plot.autoscale(False)
        #plot.set_aspect('equal', 'box')

    index += 1
    return cs
    #plot.axis('equal', 'box')


def plot_classification(X, y, plot=None, model=None, colors=None):
    """
    Plots classification of data X with y
    Classifications on plots are determined by colors with default: ['blue', 'orange']
    model
    """
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
