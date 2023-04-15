import configparser
import copy
import json
import os
import warnings
from datetime import datetime

import fire
import numpy as np
import ray
import torch
from torch import nn
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel

import loaders.cifar10
import loaders.visdrone
import loaders.widar
import loaders.wisdm
import loaders.ut_har
import wandb
from aggregators.base import FederatedAveraging
from loaders.visdrone import YOLO_HYPERPARAMETERS
from models.wisdm import LSTM_NET
from models.ut_har import *
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.user_index import UserPartition
from trainers.distributed_base import DistributedTrainer

os.environ['WANDB_START_METHOD'] = 'thread'

config = configparser.ConfigParser()
config.read('config.yml')

ray.init()


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


class Experiment:
    # def __init__(self, cfg):
    #     print(f'reading config from {cfg}')
    #     config.read(cfg)
    #     print(config['DEFAULT'].get('partition_type'))

    def main(self,
             model: str = [config['DEFAULT'].get('model', 'models/resnet_group_norm.pt'), print(config['DEFAULT'])][0],
             dataset_name: str = config['DEFAULT'].get('dataset', 'cifar10'),
             data_dir: str = config['DEFAULT'].get('data_dir', '../data/'),
             client_num_in_total: int = config['DEFAULT'].getint('client_num_in_total', 2118),
             client_num_per_round: int = config['DEFAULT'].getint('client_num_per_round', 10),
             gpu_worker_num: int = config['DEFAULT'].getint('gpu_worker_num', 8),
             batch_size: int = config['DEFAULT'].getint('batch_size', 16),
             client_optimizer: str = config['DEFAULT'].get('client_optimizer', 'sgd'),
             lr: float = config['DEFAULT'].getfloat('lr', 0.1e-2),
             wd: float = config['DEFAULT'].getfloat('wd', 0.001),
             epochs: int = config['DEFAULT'].getint('epochs', 1),
             fl_algorithm: str = config['DEFAULT'].get('fl_algorithm', 'FedAvgSeq'),
             comm_round: int = config['DEFAULT'].getint('comm_round', 30),
             test_frequency: int = config['DEFAULT'].getint('test_frequency', 2),
             server_optimizer: str = config['DEFAULT'].get('server_optimizer', 'adam'),
             server_lr: float = config['DEFAULT'].getfloat('server_lr', 1e-1),
             alpha: float = config['DEFAULT'].getfloat('alpha', 0.1),
             partition_type: str = config['DEFAULT'].get('partition_type', 'dirichlet'),
             trainer: str = config['DEFAULT'].get('trainer', 'BaseTrainer')):
        """
        :param model: neural network used in training
        :param dataset_name: dataset used for training
        :param data_dir: data directory
        :param client_num_in_total: number of workers in a distributed cluster
        :param client_num_per_round: number of workers
        :param gpu_worker_num: total GPU number
        :param batch_size: input batch size for training
        :param client_optimizer: SGD with momentum; adam
        :param lr: learning rate
        :param wd: weight decay parameter
        :param epochs: how many epochs will be trained locally
        :param fl_algorithm: Algorithm list: FedAvg; FedOPT; FedProx; FedAvgSeq
        :param comm_round: how many round of communications we should use
        :param test_frequency: the frequency of the algorithms
        :param server_optimizer: server_optimizer
        :param server_lr: server_lr
        :param alpha: alpha in Dirichlet distribution
        :param partition_type: partition type: user, dirichlet, central
        :param trainer: trainer to be used
        """

        args = copy.deepcopy(locals())
        args.pop('self')

        device = config['DEFAULT']['device']
        set_seed(1)
        print(json.dumps(args, sort_keys=True, indent=4))
        print(config['DEFAULT'].get('partition_type'))

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
        elif dataset_name == 'ut_har':
            dataset = loaders.ut_har.load_dataset()
            num_classes = 7
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
            # mode='disabled',
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
        data_ref = ray.put(client_datasets[-1].dataset)
        client_dataset_refs = [{'dataset': data_ref, 'indices': ray.put(client_dataset.indices)} for client_dataset in
                               client_datasets]
        # client_dataset_refs = [ray.put(client_dataset) for client_dataset in client_datasets]
        if trainer == 'BaseTrainer':
            global_model = torch.load(model)
            from scorers.classification_evaluator import evaluate
            scheduler = torch.optim.lr_scheduler.MultiStepLR(torch.optim.SGD(global_model.parameters(), lr=lr),
                                                             milestones=[150, 300],
                                                             gamma=0.1)
            client_trainers = [DistributedTrainer.remote(model_path=model,
                                                         state_dict=global_model.state_dict(),
                                                         criterion=nn.CrossEntropyLoss(),
                                                         optimizer_name=client_optimizer,
                                                         epochs=epochs, scheduler='multisteplr',
                                                         **{'lr': lr, 'milestones': [300, 500], 'gamma': 0.1}) for _ \
                               in range(client_num_per_round)]
        elif trainer == 'ultralytics':
            global_model = DetectionModel(cfg=model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(global_model.optimizer, T_0=10, T_mult=2,
                                                                             eta_min=1e-6)

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
            remote_steps = []

            # Select random clients for each round
            sampled_clients_idx = np.random.choice(len(client_datasets), client_num_per_round, replace=False)

            for i, client_trainer in enumerate(client_trainers):
                # Update the remote client_trainer with the latest global model and scheduler state
                client_trainer.update.remote(global_model.state_dict(), scheduler.state_dict())

                # Perform a remote training step on the client_trainer
                remote_step = client_trainer.step.remote(sampled_clients_idx[i],
                                                         client_dataset_refs[sampled_clients_idx[i]],
                                                         round_idx,
                                                         device=device)
                remote_steps.append(remote_step)

            # Retrieve remote steps results
            updates, weights, local_metrics = zip(*ray.get(remote_steps))

            # Calculate the average local metrics
            local_metrics_avg = {key: sum(d[key] for d in local_metrics) / len(local_metrics) for key in
                                 local_metrics[0]}

            # Update the global model using the aggregator
            state_n = aggregator.step(updates, weights, round_idx)
            global_model.load_state_dict(state_n)
            scheduler.step()
            wandb.log(local_metrics_avg, step=round_idx)
            if round_idx % test_frequency == 0:
                metrics = evaluate(global_model, dataset['test'], device=device, num_classes=num_classes)
                for k, v in metrics.items():
                    if type(v) == torch.Tensor:
                        v = v.item()
                    wandb.log({k: v}, step=round_idx)
                    print(f'metric round_idx = {k}: {v}')


if __name__ == '__main__':
    fire.Fire(Experiment)
