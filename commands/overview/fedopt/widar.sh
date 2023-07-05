#!/bin/bash


for seed in {1..3}
do
    ## 3. widar
    server_lr=0.01
    client_lr=0.001
    ## NIID-0.1 Adam 10%-30%
   seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 300 --batch_size 8 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ## NIID-0.5 Adam 10%-30%
    server_lr=0.01
    client_lr=0.01
   seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 300 --batch_size 8 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
done