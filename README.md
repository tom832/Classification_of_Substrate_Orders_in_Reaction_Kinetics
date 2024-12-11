# Classification_of_Substrate_Orders_in_Reaction_Kinetics

## Overview
![overview of the workflow](./assets/overview.png)

This repository contains the source data and codes for paper "Mechanistic Studies on Condensation Polymerization of C3 Propargylic Electrophiles". The paper is currently under review.

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)

- [Training and Evaluation on Test data](#training-and-evaluation-on-test-data)



## Installation

1. Git clone the repository
    - `git clone https://github.com/tom832/Classification_of_Substrate_Orders_in_Reaction_Kinetics.git`
    - `cd Classification_of_Substrate_Orders_in_Reaction_Kinetics`

2. Create conda environment from environment.yml （Recommended operating on a Linux machine）
    - `conda create -f environment.yml`
    - `conda activate cosork` stands for **C**lassification **O**f **S**ubstrate **O**rders in **R**eaction **K**inetics

3. Download and unzip the data and model
    - `bash download_data_and_model.sh`
    - you will see the following structure
    ```
    .
    ├── data
    │   ├── 5_class
    │   │   ├── raw (directory for raw data of each class)
    │   │   └── train_val.csv
    │   │   └── train_val_tsfresh_feat.csv
    │   │   └── test.csv
    │   │   └── test_tsfresh_feat.csv
    │   ├── 6_class
    │   │   ├── raw (directory for raw data of each class)
    │   │   └── train_val.csv
    │   │   └── train_val_tsfresh_feat.csv
    │   │   └── test.csv
    │   │   └── test_tsfresh_feat.csv
    │   ├── ode_raw_data (directory for raw data of ODE)
    └── model
        └── 5_class
        │   └── tsfresh_raw_feat__s__best_quality_3h (best model used in research)
        └── 6_class
            └── tsfresh_raw_feat__s__best_quality_3h (best model used in research)

    ```

## Training and Evaluation on Test data
- `python ag_train.py` to train the model
- `python ag_train.py -h` to see the help message about arguments
