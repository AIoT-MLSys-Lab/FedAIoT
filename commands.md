# Baselines

## 1. wisdm phone

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 200 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_phone --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



## 2. wisdm watch

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 200 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name wisdm_watch --model LSTM_NET --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy





## 3. widar

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 120 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name widar --model Widar_ResNet18  --client_num_in_total 40 --client_num_per_round 12 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy





## 4. ut_har

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 500 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 2 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 2 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 2 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 4 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.001 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 2 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 4 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name ut_har --model UT_HAR_ResNet18  --client_num_in_total 20 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.001 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy





## 5. emognition

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 150 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 7 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 14 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 21 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 7 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 14 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 21 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 7 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 14 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 21 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 7 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 14 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name emognition --model LSTMRegressor  --client_num_in_total 70 --client_num_per_round 21 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy





## 6. casas

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 250 --batch_size 128 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 12 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 18 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 12 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 18 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 12 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 18 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 12 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main  --dataset_name casas --model BiLSTMModel  --client_num_in_total 60 --client_num_per_round 18 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.1 --test_frequency 5 --comm_round 400 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy





## 7. energy

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.0001 --client_optimizer adam --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 1200 --batch_size 32 --analysis baseline --trainer BaseTrainer --watch_metric R^2



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer sgd --server_lr 1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.1 --lr 0.01 --server_optimizer adam --server_lr .1 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 8 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 16 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name energy --model MLP  --client_num_in_total 80 --client_num_per_round 24 --partition_type dirichlet --alpha 0.5 --lr 0.01 --server_optimizer adam --server_lr 0.01 --test_frequency 5 --comm_round 3000 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric R^2







## 8. visdrone

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness



###num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 3 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness NIID-0.1 SGD 10%-30%



num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 9 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 3 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 9 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 3 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 6 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 9 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 3 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 6 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name visdrone --model models/yolov8n.yaml --client_num_in_total 30 --client_num_per_round 9 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 600 --batch_size 12 --analysis baseline --trainer ultralytics --amp --watch_metric fitness





## 9. epic_sounds

### Centralized

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=7 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 1 --client_num_per_round 1 --partition_type central --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 10 --comm_round 150 --batch_size 512 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 32 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 60 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 90 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 SGD 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 60 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 90 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer sgd --server_lr 1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.1 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 60 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 90 --partition_type dirichlet --alpha 0.1 --lr 0.1 --server_optimizer adam --server_lr .1 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy



### NIID-0.5 Adam 10%-30%

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 30 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 60 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy

num_gpus=1 num_trainers_per_gpu=1 CUDA_VISIBLE_DEVICES=0 python distributed_main.py main --dataset_name epic_sounds --model resnet18 --client_num_in_total 300 --client_num_per_round 90 --partition_type dirichlet --alpha 0.5 --lr 0.1 --server_optimizer adam --server_lr 0.01 --test_frequency 20 --comm_round 300 --batch_size 12 --analysis baseline --trainer BaseTrainer --amp --watch_metric accuracy