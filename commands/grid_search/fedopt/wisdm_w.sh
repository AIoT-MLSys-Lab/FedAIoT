lrs=(0.01 0.001 0.0001)
for client_lr in "${lrs[@]}"
do
  for server_lr in "${lrs[@]}"
  do
    ### NIID-0.1 Adam 10%
     num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr $client_lr --server_optimizer adam --server_lr $server_lr --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
    ### NIID-0.5 Adam 10%
     num_gpus=1 num_trainers_per_gpu=8 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr $client_lr --server_optimizer adam --server_lr $server_lr --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy
  done
done