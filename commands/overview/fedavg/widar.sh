#!/bin/bash

client_lr=0.005
for seed in {1..3}
do
    ## 3. widar
    ### Centralized
#    seed=$seed num_gpus=1 num_trainers_per_gpu=1 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 120 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ### NIID-0.1 SGD 10%-30%
#    seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    seed=$seed num_gpus=3 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1500 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ### NIID-0.5 SGD 10%-30%
#    seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
#    seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ### NIID-0.1 Adam 10%-30%
#    seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ### NIID-0.5 Adam 10%-30%
#    seed=$seed num_gpus=1 num_trainers_per_gpu=4 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
done