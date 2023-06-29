#!/bin/bash

for seed in {1..3}
do
## 6. casas
## NIID-0.1 Adam 10%-30%
client_lr=0.001
server_lr=0.01
seed=$seed num_gpus=1 num_trainers_per_gpu=6 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
## NIID-0.5 Adam 10%-30%
client_lr=0.0001
server_lr=0.01
seed=$seed num_gpus=1 num_trainers_per_gpu=6 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
done