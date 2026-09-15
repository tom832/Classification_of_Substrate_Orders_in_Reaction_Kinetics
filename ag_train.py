#!/usr/bin/env python3
"""Train COSORK from data/{4,5}_class; see ag_train.sh for examples."""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent


def load_split(directory, split, feat, sp_mode, class_num):
    """Read only requested inputs, align by id, and exclude labeling metadata."""
    frames = []
    for kind in (['raw', 'tsfresh'] if feat == 'tsfresh_raw' else [feat]):
        path = directory / (f'{split}.csv' if kind == 'raw' else f'{split}_tsfresh_feat.csv')
        df = pd.read_csv(path, float_precision='round_trip')
        if not {'id', 'class'} <= set(df.columns):
            raise ValueError(f'{path}: missing id/class')
        if df['id'].isna().any() or not df['id'].is_unique:
            raise ValueError(f'{path}: missing or duplicate ids')
        expected = {f'S{i}' for i in range(1, class_num + 1)}
        if set(df['class']) != expected:
            raise ValueError(f'{path}: expected exactly {sorted(expected)}')
        if kind == 'raw':
            columns = [f'{species}{i}' for species in sp_mode for i in range(30)]
        else:
            columns = [c for c in df.columns if any(c.startswith(f'{s}__') for s in sp_mode)]
            if any(not any(c.startswith(f'{s}__') for c in columns) for s in sp_mode):
                raise ValueError(f'{path}: missing requested tsfresh series')
        frames.append(df[['id', 'class'] + columns].set_index('id'))
    result = frames[0]
    if len(frames) == 2:
        other = frames[1]
        if set(result.index) != set(other.index):
            raise ValueError(f'{split}: raw/tsfresh ids differ')
        other = other.reindex(result.index)
        if not result['class'].equals(other['class']):
            raise ValueError(f'{split}: raw/tsfresh labels differ')
        result = result.join(other.drop(columns='class'), validate='one_to_one')
    result = result.reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def evaluate(predictor, test, classes, model_dir):
    y = test['class']
    x = test.drop(columns='class')
    pred = predictor.predict(x)
    proba = predictor.predict_proba(x).reindex(columns=classes)
    scores = dict(Accuracy=accuracy_score(y, pred),
                  ROC_AUC=roc_auc_score(y, proba, labels=classes, multi_class='ovr'),
                  MCC=matthews_corrcoef(y, pred),
                  F1=f1_score(y, pred, average='weighted', zero_division=0),
                  Precision=precision_score(y, pred, average='weighted', zero_division=0),
                  Recall=recall_score(y, pred, average='weighted', zero_division=0))
    for key, value in scores.items():
        print(f'{key}: {value:.4f}', flush=True)
    print(classification_report(y, pred, labels=classes, zero_division=0))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='white')
    sns.set_context('paper', font_scale=1.5)
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    sns.heatmap(confusion_matrix(y, pred, labels=classes), annot=True, fmt='d',
                cmap='Blues', cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title='Test-set confusion matrix')
    ax.tick_params(axis='both', rotation=0, direction='out', bottom=True, left=True,
                   length=4, width=1, color='black', labelcolor='black')
    fig.tight_layout()
    fig.savefig(model_dir / 'confusion_matrix.png', bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--class_num', type=int, choices=(4, 5), default=4)
    p.add_argument('--sp_mode', choices=('s', 'p', 'sp'), default='s')
    p.add_argument('--feat', choices=('raw', 'tsfresh', 'tsfresh_raw'), default='tsfresh_raw')
    p.add_argument('--ag_train_quality', choices=('best_quality', 'high_quality',
                   'good_quality', 'medium_quality'), default='best_quality')
    p.add_argument('--hours', type=float, default=3.0)
    p.add_argument('--num_cpus', type=int, default=0, help='0 = all available CPUs')
    p.add_argument('--num_gpus', type=int, choices=(0, 1), default=0)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--verbose', type=int, choices=range(5), default=2)
    p.add_argument('--evaluate_on_test_data', action='store_true')
    p.add_argument('--data_dir', type=Path, help='optional override of data/{class_num}_class')
    p.add_argument('--model_dir', type=Path, help='optional NEW model directory')
    args = p.parse_args()
    if not np.isfinite(args.hours) or args.hours <= 0 or args.num_cpus < 0 or args.random_seed < 0:
        p.error('hours must be finite and positive; num_cpus and random_seed must be nonnegative')
    data_dir = (args.data_dir or ROOT / 'data' / f'{args.class_num}_class').resolve()
    model_dir = (args.model_dir or ROOT / 'AutogluonModels' / f'{args.class_num}_class' /
                 f'{args.feat}_feat__{args.sp_mode}__{args.ag_train_quality}__{args.hours}h').resolve()
    if model_dir.exists():
        p.error(f'Model directory already exists: {model_dir}. Use --model_dir with a new path.')
    train = load_split(data_dir, 'train_val', args.feat, args.sp_mode, args.class_num)
    test = None
    if args.evaluate_on_test_data:
        test = load_split(data_dir, 'test', args.feat, args.sp_mode, args.class_num)
        if set(train['id']) & set(test['id']):
            raise ValueError('Train/test ids overlap')
        if set(train.columns) != set(test.columns):
            raise ValueError('Train/test feature schemas differ')
        test = test.reindex(columns=train.columns)

    from autogluon.tabular import TabularPredictor
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    try:
        import torch
        torch.manual_seed(args.random_seed)
        if args.num_gpus and not torch.cuda.is_available():
            print('CUDA unavailable; using CPU.', flush=True)
            args.num_gpus = 0
    except ImportError:
        args.num_gpus = 0
    train = train.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    print(f'Data: {data_dir}\nModel: {model_dir}\nTraining shape: {train.shape}', flush=True)
    print(train['class'].value_counts().sort_index(), flush=True)
    start = time.monotonic()
    predictor = TabularPredictor(
        problem_type='multiclass', label='class', eval_metric='accuracy',
        learner_kwargs={'ignored_columns': ['id']}, path=str(model_dir),
        verbosity=args.verbose, sample_weight='balance_weight',
    ).fit(
        train_data=train, presets=args.ag_train_quality, time_limit=args.hours * 3600,
        num_cpus=args.num_cpus if args.num_cpus else 'auto',
        ag_args_fit={'num_gpus': args.num_gpus},
    )
    if test is not None:
        evaluate(predictor, test, [f'S{i}' for i in range(1, args.class_num+1)], model_dir)
    print(f'Finished in {(time.monotonic()-start)/60:.2f} min: {model_dir}', flush=True)


if __name__ == '__main__':
    main()
