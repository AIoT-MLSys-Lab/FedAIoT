import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LRScheduler

import loaders.casas
import loaders.cifar10
import loaders.ego4d
import loaders.emognition
import loaders.energy
import loaders.epic_sounds
import loaders.spatial_transforms
import loaders.ut_har
import loaders.visdrone
import loaders.widar
import loaders.wisdm
import wandb
from analyses.noise import inject_label_noise_with_matrix
from loaders.utils import ParameterDict
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.user_index import UserPartition
from partition.utils import compute_client_data_distribution, get_html_plots


def read_system_variable(system_config, ):
    num_gpus = int(os.environ['num_gpus']) if 'num_gpus' in os.environ \
        else system_config['DEFAULT'].getint('num_gpus', 1)
    num_trainers_per_gpu = int(os.environ['num_trainers_per_gpu']) if 'num_gpus' in os.environ \
        else system_config['DEFAULT'].getint('num_trainers_per_gpu', 1)
    seed = int(os.environ['seed']) if 'seed' in os.environ \
        else system_config['DEFAULT'].getint('seed', 1)
    return num_gpus, num_trainers_per_gpu, seed


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, scheduler):
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        super(WarmupScheduler, self).__init__(optimizer, -1)
        self._last_lr = [0.0] * len(optimizer.param_groups)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
            new_lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
                param_group['lr'] = lr
            self._last_lr = new_lrs
        else:
            self.scheduler.step(epoch)
            self._last_lr = self.scheduler.get_last_lr()


def get_default_yolo_hyperparameters():
    YOLO_HYPERPARAMETERS = {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'fl_gamma': 0.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'mask_ratio': 0.0,
        'overlap_mask': 0.0,
        'conf': 0.25,
        'iou': 0.45,
        'max_det': 1000,
        'plots': False,
        'half': False,  # use half precision (FP16)
        'dnn': False,
        'data': None,
        'imgsz': 640,
        'verbose': False
    }
    YOLO_HYPERPARAMETERS = ParameterDict(YOLO_HYPERPARAMETERS)
    return YOLO_HYPERPARAMETERS


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


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dataset = loaders.cifar10.load_dataset()
        num_classes = 10
    elif dataset_name == 'wisdm_watch':
        dataset = loaders.wisdm.load_dataset(reprocess=False, modality='watch')
        num_classes = 12
    elif dataset_name == 'wisdm_phone':
        dataset = loaders.wisdm.load_dataset(reprocess=False, modality='phone')
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
    elif dataset_name == 'emognition':
        dataset = loaders.emognition.load_bracelet_data(reprocess=True)
        num_classes = 2
    elif dataset_name == 'casas':
        dataset = loaders.casas.load_dataset()
        num_classes = 12
    elif dataset_name == 'energy':
        dataset = loaders.energy.load_dataset()
        num_classes = 10
    elif dataset_name == 'epic_sounds':
        dataset = loaders.epic_sounds.load_dataset()
        num_classes = 44
    elif dataset_name == 'ego4d':
        dataset = loaders.ego4d.load_dataset(
            transforms=loaders.spatial_transforms.Compose(
                [loaders.spatial_transforms.Normalize([0.45], [0.225])]
            )
        )
        num_classes = 17
        # print(dataset['train'][1][1].shape)
        # print(np.unique(dataset['train'].targets), len(np.unique(dataset['train'].targets)))
        # raise ValueError('ego4d')
    else:
        raise ValueError(f'Dataset {dataset_name} type not supported')

    return dataset, num_classes


def get_partition(partition_type, dataset_name, num_classes, client_num_in_total, client_num_per_round, alpha, dataset):
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
        raise ValueError(f'Partition {partition_type} type not supported')

    return partition, client_num_in_total, client_num_per_round


def plot_data_distributions(dataset, dataset_name, client_datasets, num_classes):
    if hasattr(dataset['train'], 'targets') and dataset_name != 'ego4d':
        data_distribution, class_distribution = compute_client_data_distribution(datasets=client_datasets,
                                                                                 num_classes=num_classes)
        class_dist, sample_dist = get_html_plots(data_distribution, class_distribution)
        wandb.log({'class_dist': wandb.Html(class_dist, inject=False),
                   'sample_dist': wandb.Html(sample_dist, inject=False)},
                  step=0)
        # if dataset_name == 'visdrone':
        #     targets = [[d['cls'] for d in dt] for dt in client_datasets]
        #     data_distribution, class_distribution = compute_client_target_distribution(targets, num_classes=12)
        #     wandb.log({'visdrone_class_dist': wandb.Html(class_dist, inject=False),
        #                'sample_dist': wandb.Html(sample_dist, inject=False)},
        #               step=0)


def add_label_noise(analysis, dataset_name, client_datasets, num_classes):
    confusion_matrix = pd.read_csv(f'confusion_matrices/conf_{dataset_name}.csv', header=0, index_col=None)
    confusion_matrix = confusion_matrix.to_numpy()
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)
    _, error_rate, error_var = analysis.split('-')
    error_rate = float(error_rate)
    error_var = float(error_var)
    print('Adding noise ...')
    client_datasets, noise_percentages = inject_label_noise_with_matrix(client_datasets,
                                                                        num_classes,
                                                                        confusion_matrix,
                                                                        error_rate)
    return client_datasets, noise_percentages


def plot_noise_distribution(noise_percentages):
    table = wandb.Table(data=[[d] for d in noise_percentages], columns=['noise_ratio'])
    wandb.log({"noise_percentages": wandb.plot.histogram(table, "noise_ratio",
                                                         title="Label Noise Distribution")
               }, step=0)
