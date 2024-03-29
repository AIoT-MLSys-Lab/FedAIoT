#!/bin/bash


client_lr=0.0001
for seed in {1..3}
do
    ## 4. ut_har
    seed=$seed num_gpus=1 num_trainers_per_gpu=2 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 2 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --precision float16 --watch_metric accuracy
 done