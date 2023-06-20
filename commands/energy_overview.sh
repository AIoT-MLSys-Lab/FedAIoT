client_lr=0.001
for seed in {1..3}
do
    ## 7. energy
    ### Centralized
    seed=$seed num_gpus=1 num_trainers_per_gpu=1 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.0001 --client_optimizer adam --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --watch_metric R^2
    ### NIID-0.1 SGD 10%-30%
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
    ### NIID-0.5 SGD 10%-30%
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
    ### NIID-0.1 Adam 10%-30%
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer adam --server_lr .1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
    ### NIID-0.5 Adam 10%-30%
    seed=$seed num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer adam --server_lr $client_lr --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2
done