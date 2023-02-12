import os

import numpy as np
# import dataframe_image as dfi

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import interpolate
from visualization import confidence_ellipse, create_colormap
from matplotlib.patches import PathPatch

"""
This module plots information about the evaluations by the separability algorithms
"""


def main():
    # column to inspect in the plots
    column = 'auc_to_maxindex'

    # datasets to include in the plots
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
        'Arrhythmia_withoutdupl_norm_05_v01',
        'Arrhythmia_withoutdupl_norm_05_v02',
        'Arrhythmia_withoutdupl_norm_05_v03',
        'Arrhythmia_withoutdupl_norm_05_v04',
        'Arrhythmia_withoutdupl_norm_05_v05',
        'Arrhythmia_withoutdupl_norm_05_v06',
        'Arrhythmia_withoutdupl_norm_05_v07',
        'Arrhythmia_withoutdupl_norm_05_v08',
        'Arrhythmia_withoutdupl_norm_05_v09',
        'Arrhythmia_withoutdupl_norm_05_v10',
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
        # 'Wilt_withoutdupl_norm_05'
    ]

    evaluation = pd.DataFrame([])
    difficulty = pd.DataFrame([])
    dataset_evaluation = pd.DataFrame([])
    auc_scores = pd.DataFrame([])

    for dataset in datasets:
        # gather IREOS index evaluations of datasets
        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0) \
                .rename(columns={f'{column}': dataset})
            evaluation = pd.concat([evaluation, results[dataset]], axis=1)
        except FileNotFoundError:
            print(f'Dataset evaluation for {dataset} not found')

        # gather difficulty evaluations on datasets
        path = os.path.join('memory', dataset, 'dataset_evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0) \
                .rename(columns={'evaluation': dataset})
            difficulty = pd.concat([difficulty, results], axis=1)
        except FileNotFoundError:
            print(f'Dataset difficulty for {dataset} not found')

    # synchronize information about all datasets
    base_path = os.path.join('memory', 'complete')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if not os.path.exists(os.path.join('plots')):
        os.makedirs(os.path.join('plots'))

    memorizing_path = os.path.join(base_path, 'evaluation.csv')
    evaluation.to_csv(memorizing_path)

    memorizing_path = os.path.join(base_path, 'difficulty.csv')
    difficulty.to_csv(memorizing_path)

    # set column identities to shorthands for the plots
    column_conv = {
        'KLR_p': 'KLR_p',
        'KLR_f': 'KLR_f',
        'LRG_nystroem_p': 'NLRG_p',
        'LRG_nystroem_f': 'NLRG_f',
        'SVM_p': 'SVM_p',
        'SVM_f': 'SVM_f',
        'KNNM_10%': 'KM_10',
        'KNNM_50%': 'KM_50',
        'KNNC_10%': 'KC_10',
        'KNNC_50%': 'KC_50',
        'MLP': 'MLP',
        'LRG_linear': 'LRG_L',
        'SVM_linear': 'SVM_L',
        'IsoForest': 'BL',
        'KNNC_W_50%': 'KCW_50',
    }

    # figure size for correlations
    plt.rcParams["figure.figsize"] = (10, 10)

    # rename columns and create transposed table
    evaluation.drop('IsoForest', inplace=True)
    evaluation.drop('KNNC_W_50%', inplace=True)
    evaluation_pd = evaluation.transpose()
    evaluation_pd.rename(columns=lambda x: column_conv[x], inplace=True)
    evaluation = evaluation_pd.transpose()

    meanpointprops = dict(marker='^', markeredgecolor='#e1165a',
                          markerfacecolor='#e1165a')
    meanlineprops = dict(linestyle='-', linewidth=2.5, color='#e1165a')

    # plot distibution of evaluations on box plots
    bp_dict = evaluation_pd.plot(kind='box', showmeans=True, color='lightblue',
                                 patch_artist=True, rot=55, fontsize=18, meanprops=meanlineprops, meanline=True)

    bp_dict.set_ylabel('Spearman correlations', fontsize=18)
    plt.show()

    plt.rcParams["figure.figsize"] = (10, 10)
    # plot agreement, i.e. equal evaluations, of separability algorithms on datasets
    ax = evaluation_pd.transpose().nunique().value_counts().plot.bar(color='skyblue', rot=0, fontsize=18)
    ax.set_yticks(np.arange(25)[::5])
    ax.set_xlabel("Number of unique values", fontsize=18)
    ax.set_ylabel("Valid for dataset variants", fontsize=18)

    plt.show()
    plt.savefig('plots/Classifier_Correlations.png')

    # show distribution of datasets in the difficulty, diversity plane with confidence ellipse
    colors_conv = {
        'Arrhythmia': '#e1165a',
        'Hepatitis': 'plum',
        'Parkinson': 'lightblue',
        'Lymphography': 'darkgoldenrod',
        'WBC': 'darkviolet',
        'Wilt': 'black',
    }

    difficulty_pd = difficulty.transpose()

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.rcParams.update({'font.size': 16})
    colors = [colors_conv[dataset.split('_')[0]] for dataset in difficulty_pd.index.to_list()]
    ax = difficulty_pd.plot.scatter(x='Difficulty', y='Diversity', c=colors)
    #ax.set_yticks(np.arange(np.ceil(difficulty_pd.Diversity.max())))
    ax.set_ylim(0, np.ceil(difficulty_pd.Diversity.max()))
    ax.set_xlim(1, np.ceil(difficulty_pd.Difficulty.max()))

    plt.savefig('plots/difficulty.png')

    # creating confidence ellipse for dataset variants
    difficulty_pd['group'] = [dataset.split('_')[0] for dataset in difficulty_pd.index.to_list()]
    for group in difficulty_pd.groupby('group').Difficulty.groups:
        dataset_properties = difficulty_pd.groupby('group')[['Difficulty', 'Diversity']].get_group(group)
        color = colors_conv[group]
        print(f'color {color}')
        confidence_ellipse(dataset_properties['Difficulty'], dataset_properties['Diversity'],
                           ax, n_std=2, edgecolor=color, group=group)

    plt.show()

    # plt.savefig('plots/difficulty.png')

    min_diff = difficulty_pd.groupby('group').Difficulty.idxmin()
    max_diff = difficulty_pd.groupby('group').Difficulty.idxmax()
    # evaluation_pd.loc[min_diff]

    # show ROC-AUC recommendations on dataset variants
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams.update({'font.size': 12})
    # auc_to_difficulty = pd.concat([difficulty_pd[['Difficulty', 'Diversity']], evaluation.mean(axis=0)], axis=1)
    auc_to_difficulty = pd.concat([difficulty_pd[['Difficulty', 'Diversity']], evaluation.mode(axis=0).transpose()[0]],
                                  axis=1)
    auc_to_difficulty.columns = ['Difficulty', 'Diversity', 'Recommended ROC-AUC by Index']
    ax = auc_to_difficulty.plot.scatter(x='Difficulty', y='Diversity', c='Recommended ROC-AUC by Index',
                                        cmap=create_colormap('e1165a', 'BFEFFF'),
                                        vmin=np.floor(auc_to_difficulty['Recommended ROC-AUC by Index'].min()),
                                        vmax=np.ceil(auc_to_difficulty['Recommended ROC-AUC by Index'].max()))
    # annotate pv8 if present
    #p8_difdiv = auc_to_difficulty.transpose()['Parkinson_withoutdupl_norm_05_v08'][['Difficulty', 'Diversity']]
    #ax.annotate('Parkinson v8', xy=(*p8_difdiv,), xycoords='data',
    #            xytext=(0.90, 0.85), textcoords='axes fraction',
    #            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8.0),
    #            horizontalalignment='right', verticalalignment='top',
    #            )

    plt.show()

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams.update({'font.size': 18})

    # show statistics of separability algorithms in bar chart
    o = pd.concat([(evaluation > evaluation.mean()).sum(axis=1).rename('Correlation above mean'),
                   (evaluation.max() == evaluation).sum(axis=1).rename('Correlation is max')],
                  axis=1).plot.bar(stacked=True, color=['#e1165a', 'lightblue'])
    o.set_ylabel('Valid for dataset variants', fontsize=18)
    plt.yticks(fontsize=18,)
    plt.xticks(fontsize=18, rotation=55,)
    plt.show()

    exit(0)


if __name__ == '__main__':
    main()
    exit(0)
