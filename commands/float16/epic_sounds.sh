CUDA_VISIBLE_DEVICES=0 seed=1 num_gpus=1 num_trainers_per_gpu=10 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --precision float16 --watch_metric accuracy
CUDA_VISIBLE_DEVICES=0 seed=2 num_gpus=1 num_trainers_per_gpu=10 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --precision float16 --watch_metric accuracy
CUDA_VISIBLE_DEVICES=0 seed=3 num_gpus=1 num_trainers_per_gpu=10 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --precision float16 --watch_metric accuracy
