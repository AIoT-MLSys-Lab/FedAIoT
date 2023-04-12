import configparser
import copy
import os
import warnings
from datetime import datetime

import click
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel
from wandb.apis.reports._panels import Vega

import loaders.cifar10
import loaders.widar
import loaders.wisdm
import loaders.visdrone
import wandb
from aggregators.base import FederatedAveraging
from models.wisdm import LSTM_NET
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.user_index import UserPartition
from partition.utils import compute_client_data_distribution, get_html_plots
from loaders.visdrone import YOLO_HYPERPARAMETERS
# from trainer import UltralyticsYoloTrainer

config = configparser.ConfigParser()
config.read('config_widar.yml')
os.environ['WANDB_START_METHOD'] = 'thread'


# ray.init()

def set_seed(seed: int):
    """
    Set the random seed for PyTorch and NumPy.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the deterministic flag for CuDNN (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@click.command()
@click.option('--model', type=click.Path(), default=config['DEFAULT'].get('model', 'models/resnet_group_norm.pt'),
              help='neural network used in training')
@click.option('--dataset_name', type=click.Choice(os.listdir('./datasets')), default=config['DEFAULT'].get('dataset',
                                                                                                           'cifar10'),
              help='dataset used for training')
@click.option('--data-dir', type=str, default=config['DEFAULT'].get('data_dir', '../data/'), help='data directory')
@click.option('--client-num-in-total', type=int, default=config['DEFAULT'].getint('client_num_in_total', 2118),
              help='number of workers in a distributed cluster')
@click.option('--client-num-per-round', type=int, default=config['DEFAULT'].getint('client_num_per_round', 10),
              help='number of workers')
@click.option('--gpu-worker-num', type=int, default=config['DEFAULT'].getint('gpu_worker_num', 8), help='total gpu num')
@click.option('--batch-size', type=int, default=config['DEFAULT'].getint('batch_size', 16),
              help='input batch size for training')
@click.option('--client-optimizer', type=str, default=config['DEFAULT'].get('client_optimizer', 'sgd'),
              help='SGD with momentum; adam')
@click.option('--lr', type=float, default=config['DEFAULT'].getfloat('lr', 0.1e-2), help='learning rate')
@click.option('--wd', type=float, default=config['DEFAULT'].getfloat('wd', 0.001), help='weight decay parameter')
@click.option('--epochs', type=int, default=config['DEFAULT'].getint('epochs', 120),
              help='how many epochs will be trained locally')
@click.option('--fl-algorithm', type=str, default=config['DEFAULT'].get('fl_algorithm', 'FedAvgSeq'),
              help='Algorithm list: FedAvg; FedOPT; FedProx; FedAvgSeq ')
@click.option('--comm-round', type=int, default=config['DEFAULT'].getint('comm_round', 30),
              help='how many round of communications we should use')
@click.option('--test_frequency', type=int, default=config['DEFAULT'].getint('test_frequency', 2),
              help='the frequency of the algorithms')
@click.option('--server-optimizer', type=str, default=config['DEFAULT'].get('server_optimizer', 'adam'),
              help='server_optimizer')
@click.option('--server-lr', type=float, default=config['DEFAULT'].getfloat('server_lr', 1e-1), help='server_lr')
@click.option('--alpha', type=float, default=config['DEFAULT'].getfloat('alpha', 0.1),
              help='alpha in direchlet distribution')
@click.option('--partition-type', type=str, default=config['DEFAULT'].get('partition_type', 'dirichlet'),
              help='partition type: user, dirichlet, central')
@click.option('--trainer', type=str, default=config['DEFAULT'].get('trainer', 'BaseTrainer'), )
def main(model,
         dataset_name,
         data_dir,
         client_num_in_total,
         client_num_per_round,
         gpu_worker_num,
         batch_size,
         client_optimizer,
         lr,
         wd,
         epochs,
         fl_algorithm,
         comm_round,
         test_frequency,
         server_optimizer,
         server_lr,
         alpha,
         partition_type,
         trainer):
    # Your function logic here...
    args = copy.deepcopy(locals())
    device = config['DEFAULT']['device']
    set_seed(1)

    if dataset_name == 'cifar10':
        dataset = loaders.cifar10.load_dataset()
        num_classes = 10
    elif dataset_name == 'wisdm':
        LSTM_NET
        dataset = loaders.wisdm.load_dataset(reprocess=False)
        num_classes = 12
    elif dataset_name == 'widar':
        dataset = loaders.widar.load_dataset()
        num_classes = 9
    elif dataset_name == 'visdrone':
        dataset = loaders.visdrone.load_dataset()
        num_classes = 12
    else:
        return
    if partition_type == 'user' and dataset_name in {'wisdm', 'widar', 'visdrone'}:
        partition = UserPartition(dataset['split']['train'])
        client_num_in_total = len(dataset['split']['train'].keys())
    elif partition_type == 'uniform':
        partition = UniformPartition(num_class=num_classes, num_clients=client_num_in_total)
    elif partition_type == 'dirichlet':
        if alpha is None:
            warnings.warn('alpha is not set, using default value 0.1')
            alpha = 0.1
        partition = DirichletPartition(num_class=num_classes, num_clients=client_num_in_total, alpha=alpha)
    elif partition_type == 'central':
        partition = CentralizedPartition()
        client_num_per_round = 1
        client_num_in_total = 1
    else:
        raise ValueError('partition type not supported')
    client_datasets = partition(dataset['train'])
    # data_distribution, class_distribution = compute_client_data_distribution(datasets=client_datasets,
    #                                                                          num_classes=num_classes)
    wandb.init(
        # mode = 'disabled',
        project='ray_fl_dev_v2',
        entity='samiul',
        name=f'{fl_algorithm}_{dataset_name}_{partition_type}_{client_num_per_round}_{client_num_in_total}_{client_optimizer}_{lr}'
             f'_{server_optimizer}_'
             f'{server_lr}_{alpha}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        config=args,
    )

    # class_dist, sample_dist = get_html_plots(data_distribution, class_distribution)
    # wandb.log({'class_dist': wandb.Html(class_dist, inject=False), 'sample_dist': wandb.Html(sample_dist, inject=False)},
    #           step=0)
    # partition = CentralizedPartition()
    # partition = DirichletPartition(client_num_in_total, alpha=alpha)

    # global_model = resnet.ResNet(resnet.BasicBlock,
    #                              [2, 2, 2, 2],
    #                              num_classes=10,
    #                              norm_layer=ResNetCustomNorm)
    if trainer == 'BaseTrainer':
        global_model = torch.load(model)
        from scorers.classification_evaluator import evaluate
        from trainers.base import BaseTrainer
        client_trainers = [BaseTrainer(model_path=model,
                                       state_dict=global_model.state_dict(),
                                       criterion=nn.CrossEntropyLoss(),
                                       optimizer_name=client_optimizer,
                                       epochs=epochs,
                                       **{'lr': lr}) for _ in range(client_num_per_round)]
    elif trainer == 'ultralytics':
        global_model = DetectionModel(cfg=model)
        global_model.args = YOLO_HYPERPARAMETERS
        from scorers.ultralytics_yolo_evaluator import evaluate
        from trainers.ultralytics import UltralyticsYoloTrainer
        client_trainers = [UltralyticsYoloTrainer(
                                       model_path=model,
                                       state_dict=global_model.state_dict(),
                                       optimizer_name=client_optimizer,
                                       epochs=epochs,
                                       device=device) for _ in range(client_num_per_round)]
    else:
        raise ValueError(f'Client trainer of type {trainer} not found')

    aggregator = FederatedAveraging(global_model=global_model,
                                    server_optimizer=server_optimizer,
                                    server_lr=server_lr,
                                    server_momentum=0.9,
                                    eps=1e-3)

    for round_idx in tqdm(range(0, comm_round)):
        updates = []
        weights = []
        for i, client_trainer in enumerate(client_trainers):
            sampled_clients_idx = np.random.choice(len(client_datasets), client_num_per_round, replace=False)
            client_trainer.update(global_model.state_dict())
            update, weight = client_trainer.step(sampled_clients_idx[i],
                                                 client_datasets[sampled_clients_idx[i]],
                                                 round_idx,
                                                 device=device)
            updates.append(update)
            weights.append(weight)

        state_n = aggregator.step(updates, weights, round_idx)
        global_model.load_state_dict(state_n)
        if round_idx % test_frequency == 0:
            metrics = evaluate(global_model, dataset['test'], device=device, num_classes=num_classes)
            for k, v in metrics.items():
                if type(v) == torch.Tensor:
                    v = v.item()
                wandb.log({k: v}, step=round_idx)
                print(f'metric round_idx = {k}: {v}')


if __name__ == '__main__':
    main()
