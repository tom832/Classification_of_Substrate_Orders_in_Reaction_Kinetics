from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os
import time
import sys
import argparse



if __name__ == '__main__':

    # load arguments
    parser = argparse.ArgumentParser(description="Training arguments for classification of substrate orders in reaction kinetics.")
    parser.add_argument('--sp_mode', type=str, default='s', help='Which concentration profile(s) to use: s, p or sp. s: substrate, p: product, sp: substrate and product. Default=s')
    parser.add_argument('--class_num', type=int, default=4, help='Number of classes: 4 or 5. Default=4')
    parser.add_argument('--feat', type=str, default='tsfresh_raw', help='Feature type: tsfresh_raw, tsfresh or raw. Default=tsfresh_raw')
    parser.add_argument('--ag_train_quality', type=str, default='best_quality', help='Autogluon train quality: best_quality, high_quality, good_quality or medium_quality. Default=best_quality')
    parser.add_argument('--hours', type=float, default=3.0, help='Training time limit in hours. Default=3.0')
    parser.add_argument('--num_cpus', type=int, default=0, help='Number of CPUs, 0 means using all CPUs. Default=0')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs: 0 or 1. Default=1')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed. Default=42')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity level: 0 to 4. Default=2')
    parser.add_argument('--evaluate_on_test_data', action='store_true', help='Evaluate on test data. Default=False')

    args = parser.parse_args()
    sp_mode = args.sp_mode
    class_num = args.class_num
    feat = args.feat
    ag_train_quality = args.ag_train_quality
    hours = args.hours
    random_seed = args.random_seed
    verbose = args.verbose
    evaluate_on_test_data = args.evaluate_on_test_data

    # check arguments
    class_list = [
        'S1',
        'S2',
        'S3',
        'S4',
        'S5',
    ]
    if class_num == 4:
        class_list.remove('S5')
    elif class_num == 5:
        pass
    else:
        print('Error: class_num must be 4 or 5.')
        sys.exit(1)
    if sp_mode not in ['s', 'p', 'sp']:
        print('Error: mode must be s, p or sp.')
        sys.exit(1)
    if feat not in ['tsfresh_raw', 'tsfresh', 'raw']:
        print('Error: feat must be tsfresh_raw, tsfresh or raw.')
        sys.exit(1)
    if args.num_gpus == 1:
        import torch
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            args.num_gpus = 0
            print('GPU not available, using CPU instead.')

    print(f'Arguments: mode={sp_mode}, class_num={class_num}, feat={feat}, ag_train_quality={ag_train_quality}, hours={hours}, evaluate_on_test_data={evaluate_on_test_data}')

    start_time = time.time()

    # load data
    print(f'Loading data for {class_num}-class classification...')
    train_val_tsfresh_data = pd.read_csv(os.path.join(f'data/', f'{str(class_num)}_class', 'train_val_tsfresh_feat.csv'))
    train_val_raw_data = pd.read_csv(os.path.join(f'data/', f'{str(class_num)}_class', 'train_val.csv'))
    

    if sp_mode == 's':
        train_val_tsfresh_data.drop(columns=[c for c in train_val_tsfresh_data.columns if c.startswith('p')], inplace=True)
        train_val_raw_data.drop(columns=[c for c in train_val_raw_data.columns if c.startswith('p')], inplace=True)
    elif sp_mode == 'p':
        train_val_tsfresh_data.drop(columns=[c for c in train_val_tsfresh_data.columns if c.startswith('s')], inplace=True)
        train_val_raw_data.drop(columns=[c for c in train_val_raw_data.columns if c.startswith('s')], inplace=True)

    if feat == 'tsfresh':
        train_val_merge_data = train_val_tsfresh_data
    elif feat == 'raw':
        train_val_merge_data = train_val_raw_data
    elif feat == 'tsfresh_raw':
        train_val_raw_data.drop(columns=['class'], inplace=True)
        train_val_merge_data = pd.merge(train_val_tsfresh_data, train_val_raw_data, on='id')

    try:
        assert train_val_merge_data.shape[0] == train_val_tsfresh_data.shape[0]
    except AssertionError:
        print(f'Error: Mismatch in number of rows between train_val_merge_data and train_val_data or test_merge_data and test_data.')
        sys.exit(1)

    # shuffle data, and replace inf with nan to avoid error in autogluon (xgboost training)
    train_val_merge_data = train_val_merge_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_val_merge_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'Loaded data shape: {train_val_merge_data.shape}')

    # train model
    label_col = 'class'
    metric = 'accuracy'
    print(f'Training on {args.num_cpus} CPUs and {args.num_gpus} GPU(s)')

    predictor = TabularPredictor(
        problem_type='multiclass',
        label=label_col,
        learner_kwargs={'ignored_columns': ['id', 'old_class']},
        path=f'./AutogluonModels/{str(class_num)}_class/{feat}_feat__{sp_mode}__{ag_train_quality}__{hours}h',
        verbosity=verbose,
        sample_weight='balance_weight',
    ).fit(
        train_data=train_val_merge_data,
        presets=ag_train_quality,
        time_limit=60*60*hours,
        ag_args_fit={"num_gpus": args.num_gpus},
        num_cpus=args.num_cpus if args.num_cpus > 0 else 'auto',
    )

    # evaluate model on test data
    if evaluate_on_test_data:
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

        print('Evaluating on test data...')
        y_pred = predictor.predict(test_merge_data_nolabel)
        y_pred_proba = predictor.predict_proba(test_merge_data_nolabel)
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'ROC_AUC: {roc_auc_score(y_test, y_pred_proba, multi_class="ovr"):.4f}')
        print(f'MCC: {matthews_corrcoef(y_test, y_pred):.4f}')
        print(f'F1: {f1_score(y_test, y_pred, average="weighted"):.4f}')
        print(f'Precision: {precision_score(y_test, y_pred, average="weighted"):.4f}')
        print(f'Recall: {recall_score(y_test, y_pred, average="weighted"):.4f}')
        print(f'Classification report:\n{classification_report(y_test, y_pred)}')

        # save confusion matrix heatmap figure
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="white")
        sns.set_context("paper", font_scale=1.5)
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(7, 7), dpi=300)
        plt.rcParams.update({'font.size': 20})
        cm = confusion_matrix(y_test, y_pred, labels=class_list)
        sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Blues, cbar=False)
        tick_marks = np.arange(len(class_list)) + 0.5
        if class_num == 5:
            class_list = ['S1', 'S2', 'S3', 'S4', 'S5']
        plt.xticks(tick_marks, class_list, rotation=0, fontsize=17)
        plt.yticks(tick_marks, class_list, rotation=0, fontsize=17)
        plt.xlabel('Predicted', fontsize=17)
        plt.ylabel('True', fontsize=17)
        plt.title(f'Confusion Matrix of Prediction on Test Data', fontsize=17)
        plt.savefig(f'./AutogluonModels/{str(class_num)}_class/{feat}_feat__{sp_mode}__{ag_train_quality}__{hours}h/confusion_matrix.png', bbox_inches='tight')
        print(f'Confusion matrix saved at ./AutogluonModels/{str(class_num)}_class/{feat}_feat__{sp_mode}__{ag_train_quality}__{hours}h/confusion_matrix.png')


    end_time = time.time()
    print(f'Elapsed time (min): {(end_time - start_time) / 60}')



