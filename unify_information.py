import os

import numpy as np
# import dataframe_image as dfi

import pandas as pd
from matplotlib import pyplot as plt


def main():
    datasets = [
        'Hepatitis_withoutdupl_norm_05_v01',
        'Hepatitis_withoutdupl_norm_05_v02',
        'Hepatitis_withoutdupl_norm_05_v03',
        'Hepatitis_withoutdupl_norm_05_v04',
        'Hepatitis_withoutdupl_norm_05_v05',
        'Hepatitis_withoutdupl_norm_05_v06',
        'Hepatitis_withoutdupl_norm_05_v07',
        'Hepatitis_withoutdupl_norm_05_v08',
        'Hepatitis_withoutdupl_norm_05_v09',
        'Hepatitis_withoutdupl_norm_05_v10',
        'Parkinson_withoutdupl_norm_05_v01',
        'Parkinson_withoutdupl_norm_05_v02',
        'Parkinson_withoutdupl_norm_05_v03',
        'Parkinson_withoutdupl_norm_05_v04',
        'Parkinson_withoutdupl_norm_05_v05',
        'Parkinson_withoutdupl_norm_05_v06',
        'Parkinson_withoutdupl_norm_05_v07',
        'Parkinson_withoutdupl_norm_05_v08',
        'Parkinson_withoutdupl_norm_05_v09',
        'Parkinson_withoutdupl_norm_05_v10',
        'WBC_withoutdupl_norm_v01',
        'WBC_withoutdupl_norm_v02',
        'WBC_withoutdupl_norm_v03',
        'WBC_withoutdupl_norm_v04',
        'WBC_withoutdupl_norm_v05',
        'WBC_withoutdupl_norm_v06',
        'WBC_withoutdupl_norm_v07',
        'WBC_withoutdupl_norm_v08',
        'WBC_withoutdupl_norm_v09',
        'WBC_withoutdupl_norm_v10',
        'Lymphography_withoutdupl_norm_1ofn',
        'Lymphography_withoutdupl_norm_idf',
        'Wilt_withoutdupl_norm_05'
    ]

    evaluation = pd.DataFrame([])
    difficulty = pd.DataFrame([])
    dataset_evaluation = pd.DataFrame([])
    auc_scores = pd.DataFrame([])

    for dataset in datasets:
        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0) \
                .rename(columns={'correlation': dataset})
            evaluation = pd.concat([evaluation, results], axis=1)
        except FileNotFoundError:
            print(f'Dataset evaluation for {dataset} not found')

        path = os.path.join('memory', dataset, 'dataset_evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0) \
                .rename(columns={'evaluation': dataset})
            difficulty = pd.concat([difficulty, results], axis=1)
        except FileNotFoundError:
            print(f'Dataset difficulty for {dataset} not found')

    memorizing_path = os.path.join('memory', 'complete', 'evaluation.csv')
    evaluation.to_csv(memorizing_path)

    memorizing_path = os.path.join('memory', 'complete', 'difficulty.csv')
    difficulty.to_csv(memorizing_path)
    # df_styled = evaluation.transpose().style.background_gradient()
    # evaluation.transpose().dfi.export("plots/evaluation.png", max_cols=-1)

    column_conv = {
        'LogisticRegression_probability': 'LG_P',
        'LogisticRegression_distance': 'LG_D',
        'KLR_probability': 'KLR_P',
        'KLR_distance': 'KLR_D',
        'SVC_probability': 'SVC_P',
        'SVC_distance': 'SVC_D',
        'KNNM_10%': 'KM_10',
        'KNNM_50%': 'KM_50',
        'KNNC_10%': 'KC_10',
        'KNNC_50%': 'KC_50',
        'LinearSVC_probability': 'LC_P',
        'LinearSVC_distance': 'LC_D',
    }

    evaluation_pd = evaluation.transpose()
    evaluation_pd.rename(columns=lambda x: column_conv[x], inplace=True)
    evaluation_pd.plot(kind='box', title='Classifier Correlations', showmeans=True, fontsize=8)
    plt.savefig('plots/Classifier_Correlations.png')

    colors_conv = {
        'Hepatitis': 'red',
        'Parkinson': 'blue',
        'Lymphography': 'orange',
        'WBC': 'green',
        'Wilt': 'black',
    }

    difficulty_pd = difficulty.transpose()
    colors = [colors_conv[dataset.split('_')[0]] for dataset in difficulty_pd.index.to_list()]
    difficulty_pd.plot.scatter(x='Difficulty', y='Diversity', c=colors)
    plt.savefig('plots/difficulty.png')

    difficulty_pd['group'] = [dataset.split('_')[0] for dataset in difficulty_pd.index.to_list()]
    min_diff = difficulty_pd.groupby('group').Difficulty.idxmin()
    evaluation_pd.loc[min_diff]

    plt.show()

    exit(0)


if __name__ == '__main__':
    main()
    exit(0)
