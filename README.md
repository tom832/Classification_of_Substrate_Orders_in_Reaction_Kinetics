# Classification_of_Substrate_Orders_in_Reaction_Kinetics

## Overview
![overview of the workflow](./assets/overview.png)

This repository contains the source data and codes for paper "Machine-Learning Assisted Mechanistic Studies on Condensation Polymerization of Propargylic Electrophiles". The paper is currently under review.

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Generate ODE data and prepare the dataset](#generate-ode-data-and-prepare-the-dataset)
- [Training COSORK Model and Evaluation on Test Data](#training-cosork-model-and-evaluation-on-test-data)
- [Analysis on the results](#analysis-on-the-results)
- [Inference on experimental data](#inference-on-experimental-data)
- [Benchmark for time-sequence deep learning models](#benchmark-for-time-sequence-deep-learning-models)
- [Bures model prediction](#bures-model-prediction)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Installation

1. Git clone the repository
    - `git clone https://github.com/tom832/Classification_of_Substrate_Orders_in_Reaction_Kinetics.git`
    - `cd Classification_of_Substrate_Orders_in_Reaction_Kinetics`

2. Create conda environment from environment.yml (Recommended operating on a Linux machine)
    - `conda env create -f environment.yml` (AutoGluon version: 1.1.1)
    - `conda activate cosork` stands for **C**lassification **O**f **S**ubstrate **O**rders in **R**eaction **K**inetics

3. Download and unzip the data and model
    - `bash download_data_and_model.sh`
    - you will see the following structure
    ```
    .
    ├── data
    │   ├── 4_class
    │   │   ├── raw (directory for raw data of each class)
    │   │   ├── train_val.csv
    │   │   ├── train_val_tsfresh_feat.csv
    │   │   ├── test.csv
    │   │   └── test_tsfresh_feat.csv
    │   ├── 5_class
    │   │   ├── raw (directory for raw data of each class)
    │   │   ├── train_val.csv
    │   │   ├── train_val_tsfresh_feat.csv
    │   │   ├── test.csv
    │   │   └── test_tsfresh_feat.csv
    │   └── ode_raw_data (directory for raw data of ODE)
    └── AutogluonModels
        ├── 4_class
        │   ├── tsfresh_raw_feat__s__best_quality_3.0h (best model used in research)
        │   ├── data_efficiency_test_result.csv
        │   ├── feature_test_result.csv
        │   └── time_test_result.csv
        └── 5_class
            ├── tsfresh_raw_feat__s__best_quality_3.0h (best model used in research)
            ├── data_efficiency_test_result.csv
            ├── feature_test_result.csv
            └── time_test_result.csv
    ```

## Generate ODE data and prepare the dataset

1. use python scripts under [`./solve_ode_scripts`](./solve_ode_scripts/) to generate ODE data (json files) in [`./data/ode_raw_data`](./data/ode_raw_data/) which has been provided in the above downloading step. Example bash script is provided in [`./solve_ode_scripts/solve_ivp_example.sh`](./solve_ode_scripts/solve_ivp_example.sh).
    ```
    .
    └── solve_ode_scripts
        ├── solve_ivp_example.sh (example script to generate 7-detailed-class ODE data)
        ├── solve_ivp_double.py
        ├── solve_ivp_single.py
        └── solve_ivp_single_outside.py
    ```

2. use notebook [`./prepare_data.ipynb`](./prepare_data.ipynb) to prepare the tabular data for training. The following steps are included:
    - Read the json raw data, transform 6 detailed classes into 4 or 5 general classes, and collect them in `./data/x_class/raw/xxx__all.csv` files
    - Stratifiedly and randomly choose 10k data for each general class as `./data/x_class/raw/xxx__10k.csv` files
    - Stratifiedly and randomly split 10% data as test data `./data/x_class/test.csv` and the rest as train and validation data `./data/x_class/train_val.csv`
    - Extract the tsfresh features as `./data/x_class/train_val_tsfresh_feat.csv` and `./data/x_class/test_tsfresh_feat.csv` files


## Training COSORK Model and Evaluation on Test Data
- `python ag_train.py`, parameters are as follows, or you can use `ag_train.sh` to train the models with different parameters.
```
>> python ag_train.py -h

Training arguments for classification of substrate orders in reaction kinetics.

options:
  -h, --help               show this help message and exit
  --sp_mode                Which concentration profile(s) to use: s, p or sp. s: substrate, p: product, sp: substrate and product. Default=s
  --class_num              Number of classes: 4 or 5. Default=4
  --feat                   Feature type: tsfresh_raw, tsfresh or raw. Default=tsfresh_raw
  --ag_train_quality       Autogluon train quality: best_quality, high_quality, good_quality or medium_quality. Default=best_quality
  --hours                  Training time limit in hours. Default=3.0
  --num_cpus               Number of CPUs, 0 means using all CPUs. Default=0
  --num_gpus               Number of GPUs: 0 or 1. Default=1
  --verbose                Verbosity level: 0 to 4. Default=2
  --evaluate_on_test_data  Evaluate on test data. Default=False
```

- Example 1: train 5-class classification model with raw features using 4 CPUs and no GPU for 0.1 hours on medium quality with verbose log to quickly check the training process, and do not evaluate on test data finally.
```
python ag_train.py \
    --sp_mode s \
    --class_num 5 \
    --feat raw \
    --ag_train_quality medium_quality \
    --hours 0.1 \
    --num_cpus 4 \
    --num_gpus 0 \
    --verbose 2
```

- Example 2: train 4-class classification model with tsfresh and raw features using 1 GPU(cuda) and all CPU for 3 hours on best quality without verbose log, and evaluate on test data finally.
```
python ag_train.py \
    --sp_mode s \
    --class_num 4 \
    --feat tsfresh_raw \
    --ag_train_quality best_quality \
    --hours 3.0 \
    --num_cpus 0 \
    --num_gpus 1 \
    --verbose 0 \
    --evaluate_on_test_data
```
Note: AutoGluon training with `best_quality` would not prefer to use GPU due to distributed training and better performance for LightGBM models.

## Analysis on the results

Jupyter notebook [`ag_result_analysis.ipynb`](./ag_result_analysis.ipynb) is provided for analysis on the results of the trained models.

Including the following for 4-class and 5-class separately:
1. Autogluon Model leaderboard on validation data
2. Evaluation and performance on test data
3. Confuxion matrix heatmap of prediction on test data
4. Chord diagram of prediction on test data
5. Probability distribution of prediction on test data (Figure 4e in the paper)
6. Benchmark for the data size (Figure S3)
7. Benchmark for the feature type
8. Benchmark for the training time

## Inference on experimental data

Jupyter notebook [`ag_experiments.ipynb`](./ag_experiments.ipynb) and wet-lab data [`./experimental_data/data.csv`](./experimental_data/data.csv) is provided for inference on experimental data and reproduce the results in our paper.

1. Prepare the experimental data
    - Read raw data
    - Normalize the time and concentration
    - Fit the concentration-time curve and obtain the 30 data points with equal time intervals to align with the model input
    - Extract the tsfresh features and concatenate with the raw features as the tabular input for Autogluon model
2. Load the trained model
3. Inference on the experimental data
4. Visualization of the prediction results (Figure 5c and 6 in the paper)
5. Single function for inference on your own experimental data with csv file or text copied from excel

    ### Inference on your own experimental data with csv file
    You can use your own experimental data with a column of `time` and a column of concentration `S` like the provided [`./experimental_data/single_data.csv`](./experimental_data/single_data.csv) to do the inference with function `predict_on_csv` in the [notebook](./ag_experiments.ipynb).

    ### Inference on your own experimental data with text copied from excel
    You can also copy the data from excel and use the function `predict_on_text` in the [notebook](./ag_experiments.ipynb) to do the inference.

NOTE: We use polynomial fitting or interpolation to align the experimental data with the model input (30 data points). You can manually choose the fitting parameters by checking their fitting plots, which are provided in the [notebook](./ag_experiments.ipynb).

## Benchmark for time-sequence deep learning models

Jupyter notebook [`benchmark_tsai.ipynb`](./benchmark_tsai.ipynb) is provided for benchmarking the time-sequence deep learning models. You can try to train and test different deep learning time-sequence models with this [notebook](./benchmark_tsai.ipynb).

Evaluation results on the test data of the trained tsai models are provided in [`./tsai_models/4_class__s/archs_results.csv`](./Tsai_Models/4_class__s/archs_results.csv), which is the same as the results Figure S4 in the paper.

## Bures model prediction

You can find conda environment file, prediction scripts and prediction results in [`./Bures_model_prediction`](./Bures_model_prediction/).

- `cd Bures_model_prediction`
- `conda create -f bures_environment.yml`
- `conda activate org_mech`
- Use [`generate_predict_analyze_data.ipynb`](./Bures_model_prediction/generate_predict_analyze_data.ipynb) to generate the data and predict the results using the Bures model and analyze the results.

## Acknowledgments
- This project uses the following awesome libraries:
    - [tsfresh](https://github.com/blue-yonder/tsfresh) for time-series feature extraction
    - [pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22) for time-series feature extraction
    - [tsai](https://github.com/timeseriesAI/tsai) for time-sequence deep learning models
    - [AutoGluon](https://github.com/autogluon/autogluon) for automatic tabular classification machine learning models
    - [seaborn](https://github.com/mwaskom/seaborn) and [matplotlib](https://github.com/matplotlib/matplotlib) for visualization

## Citation
 
 To be updated.