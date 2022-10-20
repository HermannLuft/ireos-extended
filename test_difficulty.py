import sys



sys.path.append('../../')
from ireos import IREOS
from data import load_campos_data

import numpy as np
from log_regression import KLR

'''
Metainformation:
gamma_max for Hepatitis full solution ist 1.3559999999999932
ireos for Hepatitis_v1: 0.8409293698883904

'''


def main():
    datasets = []
    # datasets with multiple variants
    #for version in range(1, 11):
    #    dataset_name = f"WBC_withoutdupl_norm_v{version:02d}"
    #    datasets.append(load_campos_data(dataset_name))
    #    print(f'{dataset_name} loaded.')



    # datasets with one variant
    dataset_name = f"Wilt_withoutdupl_norm_05"
    datasets.append(load_campos_data(dataset_name))
    print(f'{dataset_name} loaded.')

    for index, dataset in enumerate(datasets):
        Ireos = IREOS(KLR)
        Ireos.E_I = 0
        Ireos.fit(*dataset, [])
        print(f'Dataset score: {Ireos.dataset_score()} for dataset version : {index + 1: 02d}')


if __name__ == '__main__':
    main()
    exit(0)
