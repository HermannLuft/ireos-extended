import os

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

    for dataset in datasets:
        path = os.path.join('memory', dataset, 'evaluation.csv')
        try:
            results = pd.read_csv(path, index_col=0) \
                .rename(columns={'correlation': dataset})
            evaluation = pd.concat([evaluation, results], axis=1)
        except FileNotFoundError:
            print(f'Dataset evaluation for {dataset} not found')

    memorizing_path = os.path.join('memory', 'complete', 'evaluation.csv')
    evaluation.to_csv(memorizing_path)

    evaluation.rename(columns=lambda x: x.split, inplace=True)
    for dataset in datasets[:-1]:
        evaluation.plot(y=dataset)
    plt.show()

    #fig, axis = plt.subplots(2, 4)
    #for dataset_i, ax in enumerate(axis):
    #    ax.plot(evaluation.iloc[dataset_i])
    print(evaluation)


if __name__ == '__main__':
    main()
    exit(0)
