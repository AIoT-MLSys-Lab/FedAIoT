import configparser
import copy
import os
import warnings
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import ray
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel

import loaders.casas
import loaders.cifar10
import loaders.ego4d
import loaders.emognition
import loaders.energy
import loaders.epic_sounds
import loaders.ut_har
import loaders.visdrone
import loaders.widar
import loaders.wisdm
import wandb
from aggregators.base import FederatedAveraging
from analyses.noise import inject_label_noise
from loaders.utils import ParameterDict, compute_client_target_distribution
from models.ut_har import *
from models.utils import load_model
from partition.centralized import CentralizedPartition
from partition.dirichlet import DirichletPartition
from partition.uniform import UniformPartition
from partition.user_index import UserPartition
from partition.utils import compute_client_data_distribution, get_html_plots
from strategies.base_fl import basic_fedavg
from trainers.distributed_base import DistributedTrainer
from trainers.ultralytics_distributed import DistributedUltralyticsYoloTrainer
from utils import WarmupScheduler

os.environ['WANDB_START_METHOD'] = 'thread'

config = configparser.ConfigParser()
config.read('config.yml')

system_config = configparser.ConfigParser()
system_config.read('system.yml')
num_gpus = os.environ['num_gpus'] if 'num_gpus' in os.environ else system_config['DEFAULT'].getint('num_gpus', 1)
num_trainers_per_gpu = os.environ['num_trainers_per_gpu'] if 'num_gpus' in os.environ else system_config[
    'DEFAULT'].getint(
    'num_trainers_per_gpu', 1)



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
             amp: bool = config['DEFAULT'].getboolean('amp', False),
             analysis: str = config['DEFAULT'].get('analysis', 'baseline'),
             trainer: str = config['DEFAULT'].get('trainer', 'BaseTrainer'),
             class_mixup: float = config['DEFAULT'].getfloat('class_mixup', 1),
             watch_metric: str = config['DEFAULT'].get('watch_metric', 'f1_score'),
             ):
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
        :param test_frequency: the frequency of the strategies
        :param server_optimizer: server_optimizer
        :param server_lr: server_lr
        :param alpha: alpha in Dirichlet distribution
        :param partition_type: partition type: user, dirichlet, central
        :param trainer: trainer to be used
        :param amp: flag for using mixed precision
        """

        args = copy.deepcopy(locals())
        args.pop('self')

        device = config['DEFAULT']['device']
        set_seed(1)
        if dataset_name == 'cifar10':
            dataset = loaders.cifar10.load_dataset()
            num_classes = 10
        elif dataset_name == 'wisdm_watch':
            dataset = loaders.wisdm.load_dataset(reprocess=True, modality='watch')
            num_classes = 12
        elif dataset_name == 'wisdm_phone':
            dataset = loaders.wisdm.load_dataset(reprocess=True, modality='phone')
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
            num_classes = 9
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
            dataset = loaders.ego4d.load_dataset()
            num_classes = 1
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
        milestones = []
        run = wandb.init(
            mode='disabled',
            project=config['DEFAULT']['project'],
            entity=config['DEFAULT']['entity'],
            name=f'{fl_algorithm}_{dataset_name}_{partition_type}_{client_num_per_round}_{client_num_in_total}_{client_optimizer}_{lr}'
                 f'_{server_optimizer}_{model}_{analysis}'
                 f'{server_lr}_{alpha}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config=args,
        )

        client_datasets = partition(dataset['train'])
        partition_name = partition_type if partition_type != 'dirichlet' else f'{partition_type}_{alpha}'
        if hasattr(dataset['train'], 'targets'):
            data_distribution, class_distribution = compute_client_data_distribution(datasets=client_datasets,
                                                                                     num_classes=num_classes)
            class_dist, sample_dist = get_html_plots(data_distribution, class_distribution)
            wandb.log({'class_dist': wandb.Html(class_dist, inject=False),
                       'sample_dist': wandb.Html(sample_dist, inject=False)},
                      step=0)
        if dataset_name == 'visdrone':
            targets = [[d['cls'] for d in dt] for dt in client_datasets]
            data_distribution, class_distribution = compute_client_target_distribution(targets, num_classes=12)
            wandb.log({'visdrone_class_dist': wandb.Html(class_dist, inject=False),
                       'sample_dist': wandb.Html(sample_dist, inject=False)},
                      step=0)

        if 'label_noise' in analysis and dataset_name in ['wisdm', 'widar', 'ut_har', 'casas']:
            _, error_rate, error_var = analysis.split('-')
            error_rate = float(error_rate)
            error_var = float(error_var)
            client_datasets, noise_percentages = inject_label_noise(client_datasets, num_classes, error_rate, error_var)
            table = wandb.Table(data=[[d] for d in noise_percentages], columns=['noise_ratio'])
            wandb.log({"noise_percentages": wandb.plot.histogram(table, "noise_ratio",
                                                                 title="Label Noise Distribution")
                       }, step=0)

        data_ref = ray.put(dataset['train'])
        client_dataset_refs = [ray.put(client_dataset) for client_dataset in
                               client_datasets]
        global_model = load_model(model_name=model, trainer=trainer, dataset_name=dataset_name)
        if trainer == 'BaseTrainer':
            from scorers.classification_evaluator import evaluate
            if dataset_name in {'energy'}:
                from scorers.regression_evaluator import evaluate
                criterion = nn.MSELoss()
            elif dataset_name in {'emognition', 'ego4d'}:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.MultiStepLR(torch.optim.SGD(global_model.parameters(), lr=lr),
                                                             milestones=milestones,
                                                             gamma=0.1)
            client_trainers = [DistributedTrainer.remote(model_name=model,
                                                         dataset_name=dataset_name,
                                                         state_dict=global_model.state_dict(),
                                                         criterion=criterion,
                                                         batch_size=batch_size,
                                                         optimizer_name=client_optimizer,
                                                         epochs=epochs, scheduler='multisteplr',
                                                         class_mixup=class_mixup,
                                                         amp=amp,
                                                         **{'lr': lr, 'milestones': milestones, 'gamma': 0.1}) for _ \
                               in range(min(client_num_per_round, num_gpus * num_trainers_per_gpu))]
        elif trainer == 'ultralytics':
            # pt = torch.load('yolov8n.pt.1')
            global_model = DetectionModel(cfg=model)
            # global_model.load(pt)
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                torch.optim.SGD(global_model.parameters(), lr=lr), T_0=10, T_mult=2,
                eta_min=1e-6)
            optimizer = torch.optim.SGD(global_model.parameters(), lr=lr)
            scheduler = WarmupScheduler(optimizer, warmup_epochs=3, scheduler=base_scheduler)
            global_model.args = YOLO_HYPERPARAMETERS
            from scorers.ultralytics_yolo_evaluator import evaluate
            client_trainers = [DistributedUltralyticsYoloTrainer.remote(
                model_path=model,
                state_dict=global_model.state_dict(),
                optimizer_name=client_optimizer,
                epochs=epochs,
                args=YOLO_HYPERPARAMETERS,
                batch_size=batch_size,
                amp=amp,
                device=device) for _ in range(client_num_per_round)]
        else:
            raise ValueError(f'Client trainer of type {trainer} not found')

        aggregator = FederatedAveraging(global_model=global_model,
                                        server_optimizer=server_optimizer,
                                        server_lr=server_lr,
                                        server_momentum=0.9,
                                        eps=1e-3)

        best_metric = -np.inf
        best_model = None

        for round_idx in tqdm(range(0, comm_round)):
            if round_idx % test_frequency == 0:
                metrics = evaluate(global_model, dataset['test'], device=device, num_classes=num_classes)
                for k, v in metrics.items():
                    if type(v) == torch.Tensor:
                        v = v.item()
                    if k == watch_metric and v > best_metric:
                        best_metric = v
                        best_model = copy.deepcopy(global_model)
                        path = f'weights/{wandb.run.name}'
                        Path(path).mkdir(parents=True, exist_ok=True)
                        torch.save(best_model.state_dict(), f'{path}/best_model.pt')
                    wandb.log({k: v}, step=round_idx)

                    print(f'metric round_idx = {k}: {v}')

            local_metrics_avg, global_model, scheduler = basic_fedavg(aggregator,
                                                                      client_trainers,
                                                                      client_dataset_refs,
                                                                      client_num_per_round,
                                                                      global_model,
                                                                      round_idx,
                                                                      scheduler,
                                                                      device, )
            print(local_metrics_avg)
            wandb.log(local_metrics_avg, step=round_idx)


if __name__ == '__main__':
    fire.Fire(Experiment)
