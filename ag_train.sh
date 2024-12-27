# python ag_train.py \
#     --sp_mode s \
#     --class_num 6 \
#     --feat tsfresh_raw \
#     --ag_train_quality best_quality \
#     --hours 3.0 \
#     --num_cpus 10 \
#     --num_gpus 1 \
#     --verbose 2 \
#     --evaluate_on_test_data

# python ag_train.py \
#     --sp_mode s \
#     --class_num 5 \
#     --feat tsfresh_raw \
#     --ag_train_quality best_quality \
#     --hours 3.0 \
#     --num_cpus 10 \
#     --num_gpus 1 \
#     --verbose 2 \
#     --evaluate_on_test_data


python ag_train.py \
    --sp_mode s \
    --class_num 6 \
    --feat tsfresh \
    --ag_train_quality best_quality \
    --hours 3.0 \
    --num_cpus 10 \
    --num_gpus 1 \
    --verbose 0 \
    --evaluate_on_test_data

python ag_train.py \
    --sp_mode s \
    --class_num 6 \
    --feat raw \
    --ag_train_quality best_quality \
    --hours 3.0 \
    --num_cpus 10 \
    --num_gpus 1 \
    --verbose 0 \
    --evaluate_on_test_data