from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

class_list = [
    'S1',
    'S2',
    'S3',
    'S4',
    'S5',
]
class_num = len(class_list)

ag_model_root = './AutogluonModels/'
sp_mode = 's'
feat = 'tsfresh_raw'
ag_train_quality = 'best_quality'
hours = 3.0
label_col = 'class'
random_seed = 42

# load test data
test_tsfresh_data = pd.read_csv(os.path.join(f'data/', f'{str(class_num)}_class', 'test_tsfresh_feat.csv'))
test_raw_data = pd.read_csv(os.path.join(f'data/', f'{str(class_num)}_class', 'test.csv'))
if sp_mode == 's':
    test_tsfresh_data.drop(columns=[c for c in test_tsfresh_data.columns if c.startswith('p')], inplace=True)
    test_raw_data.drop(columns=[c for c in test_raw_data.columns if c.startswith('p')], inplace=True)
elif sp_mode == 'p':
    test_tsfresh_data.drop(columns=[c for c in test_tsfresh_data.columns if c.startswith('s')], inplace=True)
    test_raw_data.drop(columns=[c for c in test_raw_data.columns if c.startswith('s')], inplace=True)
if feat == 'tsfresh':
    test_merge_data = test_tsfresh_data
elif feat == 'raw':
    test_merge_data = test_raw_data
elif feat == 'tsfresh_raw':
    test_raw_data.drop(columns=['class'], inplace=True)
    test_merge_data = pd.merge(test_tsfresh_data, test_raw_data, on='id')
test_merge_data.replace([np.inf, -np.inf], np.nan, inplace=True)
y_test = test_merge_data[label_col]
test_merge_data_nolabel = test_merge_data.drop(columns=[label_col])  # delete label column

# load model
model_root = os.path.join(ag_model_root, f'{str(class_num)}_class', f'{feat}_feat__{sp_mode}__{ag_train_quality}__{hours}h')
predictor = TabularPredictor.load(model_root)


# this may take 20 hours
df_feature_importance_best = predictor.feature_importance(
    data = test_merge_data,
    subsample_size = 5000,
    num_shuffle_sets = 10,
    time_limit = 10*60*60,
)
df_feature_importance_best.to_csv(f'feature_importance_{len(class_list)}class_5000subsample_10shuffle.csv')