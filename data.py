import os
import pandas as pd
import numpy as np
from scipy.stats import ortho_group
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.io import arff


def get_univariate_schaeffler_X_y():
    file_path = '../../data/noise_data_dict.pkl'
    raw_data = pd.read_pickle(file_path)

    UNIVARIATE_VARS = [
        "Effektivwert",
        "TB_Effektiv",
        "MB_Effektiv",
        "HB_Effektiv",
        "VHB_Effektiv",
        "Crestfaktor",
    ]
    # NOTE: "Geschnitten" not added

    data = np.zeros((len(UNIVARIATE_VARS), len(raw_data[UNIVARIATE_VARS[0]])))
    for i, var in enumerate(UNIVARIATE_VARS):
        data[i] = raw_data[var]
    X = data.T
    y = raw_data["Label_NOK"]  # 0 == OK, 1 == NOK

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y


def get_schaeffler_pxx(low=True):
    # normal labels
    file_path = '../../data/noise_data_dict.pkl'
    raw_data = pd.read_pickle(file_path)
    if low:
        X = raw_data["Pxx_Werte_Low"]
    else:
        X = raw_data["Pxx_Werte_Hi"]
    y = raw_data["Label_NOK"]

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y


def get_schaeffler_pxx_normal_and_expert(low=True):
    # normal labels
    file_path = '../../data/noise_data_dict.pkl'
    raw_data = pd.read_pickle(file_path)
    if low:
        X = raw_data["Pxx_Werte_Low"]
    else:
        X = raw_data["Pxx_Werte_Hi"]
    y = raw_data["Label_NOK"]

    # expert labels
    expert_file_path = '../../data/ground_df_3cl_classified.pkl'
    expert_raw_data = pd.read_pickle(expert_file_path)
    expert_label_indices = np.array(expert_raw_data.index)

    # cut away samples from ml dataset with gt labels
    expert_X = X[expert_raw_data.index]
    X = np.delete(X, expert_label_indices, axis=0)
    y = np.delete(y, expert_label_indices, axis=0)

    scaler = MinMaxScaler()

    # fit on ml data
    X_norm = scaler.fit_transform(X)
    # only transformer on gt data
    expert_X_norm = scaler.transform(expert_X)

    expert_y = expert_raw_data["Expert_Labels"].to_numpy()  # 0 == OK, 1 == NOK
    return X_norm, y, expert_X_norm, expert_y


def get_schaeffler_expert_labels_X_y(uncertain_is_inlier=True):
    file_path = '../../data/ground_df_3cl_classified.pkl'
    UNIVARIATE_VARS = [
        "Effektivwert",
        "TB_Effektiv",
        "MB_Effektiv",
        "HB_Effektiv",
        "VHB_Effektiv",
        "Crestfaktor",
        "Max_Average",
    ]
    raw_data = pd.read_pickle(file_path)

    data = np.zeros((len(UNIVARIATE_VARS), len(raw_data[UNIVARIATE_VARS[0]])))
    for i, var in enumerate(UNIVARIATE_VARS):
        data[i] = raw_data[var]
    X = data.T
    y = raw_data["Expert_Labels"].to_numpy()  # 0 == OK, 1 == NOK
    if uncertain_is_inlier:
        y[y == 0.5] = 0

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    index = raw_data.index.to_numpy()
    y_old = raw_data["Machine_Label"].to_numpy()
    return X_norm, y, index, y_old


def get_parkinson_X_y():
    dir_name = os.path.dirname(__file__)
    path_to_dataset = os.path.join(dir_name, "datasets/parkinsons.data")
    data = pd.read_csv(path_to_dataset).drop(["name"], axis=1)
    scaler = MinMaxScaler()
    y = data["status"]
    X = data.drop(["status"], axis=1)

    X.iloc[:, :] = scaler.fit_transform(X)
    return X, y


def get_dataset_prepared(name, y_column, id):
    dir_name = os.path.dirname(__file__)
    path_to_dataset = os.path.join(dir_name, f"datasets/{name}.data")
    data = pd.read_csv(path_to_dataset).drop([id], axis=1)
    pipe = Pipeline([('scaler', StandardScaler()), ('normalizer', MinMaxScaler())])
    y = data[y_column]
    X = data.drop([y_column], axis=1)

    X.iloc[:, :] = pipe.fit_transform(X)
    return X.to_numpy(), y


def load_campos_data(name):
    dir_name = os.path.dirname(__file__)
    path_to_dataset = os.path.join(dir_name, f"datasets/{name.split('_')[0]}/{name}.arff")

    data = arff.loadarff(path_to_dataset)
    data = pd.DataFrame(data[0]).drop("id", axis=1)

    y = data["outlier"].map({
        b'yes': 1,
        b'no': 0
    })
    X = data.drop(["outlier"], axis=1)

    return X.to_numpy(), y


def get_synthetic_features(n_dims, cluster_distance, n_samples, contamination):
    """
    Get a dataset with samples drawn from a multivariate gaussian. The last mean and covariance is that of the outlier
    class. If more than two means/covariance matrices are provided then each class is interpreted as inlier class
    and outlier samples are distributed equally to all clusters.
    """
    means = np.array([
        np.zeros(n_dims),  # mean of inlier samples
        np.ones(n_dims) * np.sqrt(cluster_distance ** 2 / n_dims),  # mean of outlier samples
    ], dtype=np.float64)
    print(f"Distance between the two clusters means: {np.linalg.norm(means[0] - means[1])}")

    cov_matrices = np.array([
        # covariance matrix of inlier class
        np.diag(np.ones(n_dims))
        ,
        # covariance matrix of outlier class
        np.diag(np.ones(n_dims)) * 2,
    ])
    n_inlier_clusters = len(means) - 1
    outlier_mean = means[-1]
    outlier_cov = cov_matrices[-1]
    n_outlier = int(contamination * n_samples)
    n_inlier = n_samples - n_outlier
    n_inlier_per_class = n_inlier // n_inlier_clusters
    assert (
            n_inlier % n_inlier_clusters == 0
    ), "cannot distribute number of samples to each class equally"

    samples = np.random.multivariate_normal(outlier_mean, outlier_cov, size=n_outlier)
    y = np.ones(n_outlier)
    for mean, cov in zip(means[:-1], cov_matrices[:-1]):
        samples = np.concatenate(
            (samples, np.random.multivariate_normal(mean, cov, size=n_inlier_per_class))
        )
        y = np.concatenate((y, np.zeros(n_inlier_per_class)))
    X = MinMaxScaler().fit_transform(samples)
    return X, y


def create_data(n_inlier=49, n_outlier=1, mean_inlier=[1, 1], mean_outlier=[1, 1]):
    assert (n_inlier + n_outlier) % 2 == 0, "Number of datapoints has to be even"

    """
        Get a dataset with samples drawn from a multivariate gaussian. 
    """

    cov_inlier = ortho_group.rvs(dim=2)
    cov_inlier = np.dot(cov_inlier, cov_inlier.T) * 5
    data_1 = np.random.multivariate_normal(mean_inlier, cov_inlier, n_inlier)

    cov_outlier = ortho_group.rvs(dim=2)
    cov_outlier = np.dot(cov_outlier, cov_outlier.T) * 5
    data_2 = np.random.multivariate_normal(mean_outlier, cov_outlier, n_outlier)

    X = np.vstack((data_1, data_2))
    Y = np.concatenate((np.zeros(shape=n_inlier), np.ones(shape=n_outlier)))

    return X, Y
