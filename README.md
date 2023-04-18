# FedAIoT: A Federated Learning Benchmark for Artificial Intelligence of Things

This repository contains the implementation of a federated learning experiments on different IoT datasets using various 
federated learning algorithms.

## Table of Contents
1. [Requirements](#requirements)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Training](#training)
5. [Usage](#usage)

## Requirements

Make sure to have the following dependencies installed:

- Python 3.6 or higher
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

## Models

The experiment supports various models and allows you to use custom models as well. See the models directory for the 
individual implementations of the models for the respective datasets.

## Training

The experiment supports different federated learning algorithms and partition types. You can configure the experiment settings by modifying the `config.yml` file or passing the required parameters when running the script.

The following federated learning algorithms are supported:

- FedAvg
- FedAdam
- FedAdaGrad
- Any combination of PyTorch optimizers at the client and server side

Supported partition types include:

- User (Natural partitioning given by the client mapping)
- Uniform
- Dirichlet
- Centralized

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

This command runs the federated learning experiment using the ResNet model with group normalization on the CIFAR-10 dataset, with 10 clients per round and 30 communication rounds.
