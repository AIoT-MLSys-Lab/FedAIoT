# FedAIoT: A Federated Learning Benchmark for Artificial Intelligence of Things

This repository contains the implementation of a federated learning experiments on different IoT datasets using various 
federated learning algorithms.

## Table of Contents
1. [Requirements](#requirements)
2. [Datasets](#datasets)
3. [Loaders](#loaders)
4. [Partitioners](#partition)
5. [Models](#models)
6. [Training](#training)
7. [Usage](#usage)

## Requirements

Make sure to have the following dependencies installed:

- Python 3.10 or higher
- PyTorch
- NumPy
- Ray
- Fire
- WandB
- Ultralytics
- ...
- 
For a full list see requirements.txt. Run:
```bash
pip install -r requirements.txt
```
## Datasets

The implemented federated learning experiment supports the following datasets:

- CIFAR-10
- WISDM
- WIDAR
- VisDrone
- UT HAR

Each dataset folder contains the `download.py` script to download the dataset.

## Loaders
Each dataset also comes with their own loaders. Based on the nature of the dataset, the loaders return a `dict` with 
specific keys. 

If the dataset has partitions for `train`, `test`, and `validation`, the loader will return the data from the respective 
partitions as PyTorch Datasets. If the dataset does not have partitions, the loader will split the data into `train`, 
`test` and `validation` partitions. In the latter case the loader will also return a `full_dataset` containing the full 
data.

If the samples in the dataset are mapped to subjects/users/clients/etc, the loaders will return a `client_mapping`. 
The loader will also return a `split` containing the client mapping for `train` and `test` partition if present.

When loading make sure to specify the classes:

```python
if dataset_name == 'cifar10':
    dataset = loaders.cifar10.load_raw_data()
    num_classes = 10
elif dataset_name == 'wisdm':
    LSTM_NET
    dataset = loaders.wisdm.load_raw_data(reprocess=False)
    num_classes = 12
elif dataset_name == 'widar':
    dataset = loaders.widar.load_raw_data()
    num_classes = 9
elif dataset_name == 'visdrone':
    dataset = loaders.visdrone.load_raw_data()
    num_classes = 12
elif dataset_name == 'ut_har':
    dataset = loaders.ut_har.load_raw_data()
    num_classes = 7
```
[See here](https://github.com/AIoT-MLSys-Lab/FedAIoT/blob/61d8147d56f7ef4ea04d43a708f4de523f9e36bc/distributed_main.py#L111-L126)

## Partition
The partition classes split a large dataset into a list of smaller datasets. Several Partition methods are implemented. 
1. Centralized essentially returns the original dataset as a list of one dataset.
2. User partitions the dataset based on the client mapping. The client mapping is a dictionary returned from the 
   corresponding dataset loader or a custom mapping created manually.
3. Uniform partitions the dataset into a specified number of clients with IID distribution.
4. Dirichlet partitions the dataset into a specified number of clients with non-IID dirichlet distribution.

Create a partition object and use that to prtition any centralized dataset. Using the same partition on two 
different data splits will result in the same distribution of data between clients. For example:
```python
    partition = DirichletPartition(num_clients=10)
    train_partition = partition(dataset['train'])
    test_partition = partition(dataset['test'])
```
Here `train_partition` and `test_partition` will have `10` clients with the same relative class and sample  
distribution.

For more implementation as has been used in code [See here](https://github.com/AIoT-MLSys-Lab/FedAIoT/blob/61d8147d56f7ef4ea04d43a708f4de523f9e36bc/distributed_main.py#L129-L145)


## Models

The experiment supports various models and allows you to use custom models as well. See the models directory for the 
individual implementations of the models for the respective datasets.

## Training

The experiment supports different federated learning algorithms and partition types. You can configure the experiment settings by modifying the `config.yml` file or passing the required parameters when running the script.

The basic federated learning algorithm is implemented in the `algorithm.base_fl` module. Given an `aggregator` (See 
aggregator module), `client_trainers` (ray actors for distributed training), `client_dataset_refs` (ray data 
references), `client_num_per_round` (Number of clients sampled per round; < total clients), `global_model`, `round_idx`, 
`scheduler`, `device` (cpu or gpu), it runs one round of federated learning following vanilla fed avg.
The following federated learning algorithms are supported:

- FedAvg
- FedAdam
- FedAdaGrad
- Any combination of PyTorch optimizers at the client and server side


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
