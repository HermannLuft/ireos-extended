# Separability algorithm benchmark for Internal, Relative Evaluation of Outlier Solutions (IREOS)
## Installation
This model requires several packages to run:


| Package         | Version |
|-----------------|---------|
| scipy           | 1.7.3   |
| pandas          | 1.4.3   |
| numpy           | 1.22.4  |
| scikit-learn    | 1.1.1   |
| pyod            | 1.0.5   |
| joblib          | 1.1.0   |
| matplotlib      | 3.5.2   |
| cvxpy           | 1.2.1   |

```sh
pip install scipy pandas numpy scikit-learn pyod joblib matplotlib cvxpy
```

## Create your environment
Environment is configured at environment.py
### Anomaly detection models
Add your detection models to the setter of the anomaly_detection function. Range may utilize function arguments kwargs provided to the environment class. 
``` python
self._anomaly_detection = [
    dict(ad_algorithm=<PyOD class>, r_name=<hyperparameter>, interval=<range>),
]
```
For example: 
``` python
dict(ad_algorithm=COF, r_name='n_neighbors', interval=(2, min(100, kwargs['n_samples']))),
```


### Separability algorithms
Add your separability algorithms to the setter of the separability algorithms function. 
Separability algorithms are provided by the scikit-learn or own implementations. Separability algorithms
must have a decision and a fit function. 
``` python
self._separability_algorithms = [
    (<name>, <model>, dict(**<ireos class args>),
]
```
For example:
``` python
self._separability_algorithms = [
    ('KLR_p', KLR, dict(r_name='gamma', metric='probability', sample_size=100, c_args=dict(kernel='rbf', C=100, ))),
]
```

## Run
Datasets must be located at ./datasets/{prefix}/{dataset name}.arff.<br />
Example dataset: Parkinson_withoutdupl_norm_05_v04 <br />
Open your favorite Terminal and run the following command:
```sh
cd <absolute path to ireos>
```
### Compute correlations
The following key indicators are computed: Spearman correlation, ROC-AUC (score; rank) recommendation, time<br />
Run the following commands:
```sh
python ./main.py <dataset name>
```
### Compute difficulty and diversity
Run the following commands:
```sh
python ./difficulty.py <dataset name>
```
### Visualize results
Results consist of distrubutions, statistics of the chosen key indicator and
outcomes for the difficulty and diversity space.
Run the following command:
```sh
python ./unify_information.py
```
### Visualize intermediate informations
Visualize outlierness assignment on dataset<br />
Run the following command:
```sh
python ./outlierness.py <dataset name>
```
Visualize separability curves on example dataset<br />
Run the following command:
```sh
python ./separability_curves.py 
```
