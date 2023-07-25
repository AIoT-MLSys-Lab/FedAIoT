#!/bin/bash


client_lr=0.01
for seed in {1..3}
do
## 6. casas
seed=$seed num_gpus=1 num_trainers_per_gpu=6 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis label_noise-0.1-0.0 --trainer BaseTrainer --amp --watch_metric accuracy
seed=$seed num_gpus=1 num_trainers_per_gpu=6 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis label_noise-0.3-0.0 --trainer BaseTrainer --amp --watch_metric accuracy
done