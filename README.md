# FedAIoT: A Federated Learning Benchmark for Artificial Intelligence of Things

This repository contains the implementation of a federated learning experiments on different IoT datasets using various 
federated learning algorithms.

## Table of Contents
1. [Requirements](#requirements)
2. [Datasets](#datasets)
3. [Partitioners](#partition)
4. [Training](#training)
5. [Usage](#usage)

## Requirements

```bash
pip install -r requirements.txt
```
## Datasets

The implemented federated learning experiment supports the following datasets:

- WISDM
- UT HAR
- WIDAR
- VisDrone
- CASAS
- AEP
- EPIC-SOUNDS

Each dataset folder contains the `download.py` script to download the dataset.

## Non-IID Partition Scheme
The partition classes split a large dataset into a list of smaller datasets. Several Partition methods are implemented. 
1. Centralized essentially returns the original dataset as a list of one dataset.
2. Dirichlet partitions the dataset into a specified number of clients with non-IID dirichlet distribution.

Create a partition object and use that to prtition any centralized dataset. Using the same partition on two 
different data splits will result in the same distribution of data between clients. For example:
```python
    partition = DirichletPartition(num_clients=10)
    train_partition = partition(dataset['train'])
```
Here `train_partition` and `test_partition` will have `10` clients with the same relative class and sample  
distribution.

For more details on implementation: [See here](https://github.com/AIoT-MLSys-Lab/FedAIoT/blob/61d8147d56f7ef4ea04d43a708f4de523f9e36bc/distributed_main.py#L129-L145)


[//]: # (## Models)

[//]: # ()
[//]: # (The experiment supports various models and allows you to use custom models as well. See the models directory for the )

[//]: # (individual implementations of the models for the respective datasets.)

## Training

The experiment supports different federated learning algorithms and partition types. You can configure the experiment settings by modifying the `config.yml` file or passing the required parameters when running the script.

The basic federated learning algorithm is implemented in the `algorithm.base_fl` module. Given an `aggregator` (See 
aggregator module), `client_trainers` (ray actors for distributed training), `client_dataset_refs` (ray data 
references), `client_num_per_round` (Number of clients sampled per round; < total clients), `global_model`, `round_idx`, 
`scheduler`, `device` (cpu or gpu), it runs one round of federated learning following vanilla fed avg.
The following federated learning algorithms are included in the benchmark:

- FedAvg
- FedAdam


Various training options and hyperparameters can be configured, such as the optimizer, learning rate, weight decay, epochs, and more.

## Usage

To run the federated learning experiment, execute the script using the following command:

```
python distributed_main.py main
```

You can also pass the required parameters when running the script. For example:

```
python distributed_main.py main --model=models/resnet_group_norm.pt --dataset_name=cifar10 --client_num_per_round=10 --comm_round=30
```
For the full list of parameters, run:
```
python distributed_main.py main --help
```
