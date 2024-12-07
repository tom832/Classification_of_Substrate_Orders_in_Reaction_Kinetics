from autogluon.tabular import TabularPredictor

import pandas as pd
import numpy as np
import os
import time
import sys


random_seed = 29


class_list = [
    'zero',
    'first',
    'second',
    'mm_1',
    # 'mm_1_out',
    'mm_2',
]


if __name__ == '__main__':

    cat_conc = sys.argv[1]
    hours = int(sys.argv[2])

    start_time = time.time()
    
    label_col = 'class'
    train_val_data_path = os.path.join(f'data/{cat_conc}', f'{cat_conc}_train_val_tsfresh_feat.csv')
    test_data_path = os.path.join(f'data/{cat_conc}', f'{cat_conc}_test_tsfresh_feat.csv')
    train_val_data = pd.read_csv(train_val_data_path)
    test_data = pd.read_csv(test_data_path)
    train_val_data.drop(columns=[c for c in train_val_data.columns if c.startswith('p')], inplace=True)
    test_data.drop(columns=[c for c in test_data.columns if c.startswith('p')], inplace=True)

    train_val_raw_data = pd.read_csv(os.path.join(f'data/{cat_conc}', f'{cat_conc}_train_val.csv'))
    test_raw_data = pd.read_csv(os.path.join(f'data/{cat_conc}', f'{cat_conc}_test.csv'))
    train_val_raw_data.drop(columns=['class'], inplace=True)
    test_raw_data.drop(columns=['class'], inplace=True)
    train_val_raw_data.drop(columns=[c for c in train_val_raw_data.columns if c.startswith('p')], inplace=True)
    test_raw_data.drop(columns=[c for c in test_raw_data.columns if c.startswith('p')], inplace=True)

    train_val_merge_data = pd.merge(train_val_data, train_val_raw_data, on='id')
    test_merge_data = pd.merge(test_data, test_raw_data, on='id')
    assert train_val_merge_data.shape[0] == train_val_data.shape[0]
    assert test_merge_data.shape[0] == test_data.shape[0]
    train_val_merge_data = train_val_merge_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_merge_data = test_merge_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_val_merge_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_merge_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    metric = 'accuracy'
    y_test = test_merge_data[label_col]
    test_merge_data_nolabel = test_merge_data.drop(columns=[label_col])  # delete label column

    predictor = TabularPredictor(
        label=label_col,
        learner_kwargs={'ignored_columns': ['id', 'old_class']},
        path=f'./AutogluonModels/{cat_conc}__tsfresh_raw_feat__s__best_quality__{hours}h',
        verbosity=0,
        sample_weight='balance_weight',
    ).fit(
        train_val_merge_data,
        num_gpus=1,
        num_cpus=4,
        presets='best_quality',
        time_limit=60*60*hours,
    )
    print(f'=========== {cat_conc} ==========')

    end_time = time.time()
    print(f'Elapsed time (min): {(end_time - start_time) / 60}')



