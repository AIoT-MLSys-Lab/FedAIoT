#!/bin/bash

client_lr=0.1
for seed in {1..3}
do
    seed=$seed num_gpus=3 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis label_noise --trainer BaseTrainer --amp --watch_metric R^2
done

