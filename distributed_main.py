import configparser
import copy
import os
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

import wandb
from aggregators.base import FederatedAveraging
from loaders.utils import get_confusion_matrix_plot
from models.ut_har import *
from models.utils import load_model
from strategies.base_fl import distributed_fedavg
from trainers.distributed_base import DistributedTrainer
from trainers.ultralytics_distributed import DistributedUltralyticsYoloTrainer
from utils import WarmupScheduler, read_system_variable, get_default_yolo_hyperparameters, set_seed, load_dataset, \
    get_partition, plot_data_distributions, add_label_noise, plot_noise_distribution

os.environ['WANDB_START_METHOD'] = 'thread'

system_config = configparser.ConfigParser()
run_config = configparser.ConfigParser()

run_config.read('config.yml')
system_config.read('system.yml')

num_gpus, num_trainers_per_gpu = read_system_variable(system_config)

YOLO_HYPERPARAMETERS = get_default_yolo_hyperparameters()

ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=num_gpus)
print("success")


class Experiment:
    # def __init__(self, cfg):
    #     print(f'reading config from {cfg}')
    #     config.read(cfg)
    #     print(config['DEFAULT'].get('partition_type'))

    def main(self,
             model: str =
             [run_config['DEFAULT'].get('model', 'models/resnet_group_norm.pt'), print(run_config['DEFAULT'])][0],
             dataset_name: str = run_config['DEFAULT'].get('dataset', 'cifar10'),
             data_dir: str = run_config['DEFAULT'].get('data_dir', '../data/'),
             client_num_in_total: int = run_config['DEFAULT'].getint('client_num_in_total', 2118),
             client_num_per_round: int = run_config['DEFAULT'].getint('client_num_per_round', 10),
             batch_size: int = run_config['DEFAULT'].getint('batch_size', 16),
             client_optimizer: str = run_config['DEFAULT'].get('client_optimizer', 'sgd'),
             lr: float = run_config['DEFAULT'].getfloat('lr', 0.1e-2),
             wd: float = run_config['DEFAULT'].getfloat('wd', 0.001),
             epochs: int = run_config['DEFAULT'].getint('epochs', 1),
             fl_algorithm: str = run_config['DEFAULT'].get('fl_algorithm', 'FedAvgSeq'),
             comm_round: int = run_config['DEFAULT'].getint('comm_round', 30),
             test_frequency: int = run_config['DEFAULT'].getint('test_frequency', 2),
             server_optimizer: str = run_config['DEFAULT'].get('server_optimizer', 'adam'),
             server_lr: float = run_config['DEFAULT'].getfloat('server_lr', 1e-1),
             alpha: float = run_config['DEFAULT'].getfloat('alpha', 0.1),
             partition_type: str = run_config['DEFAULT'].get('partition_type', 'dirichlet'),
             amp: bool = run_config['DEFAULT'].getboolean('amp', False),
             analysis: str = run_config['DEFAULT'].get('analysis', 'baseline'),
             trainer: str = run_config['DEFAULT'].get('trainer', 'BaseTrainer'),
             class_mixup: float = run_config['DEFAULT'].getfloat('class_mixup', 1),
             precision: str = run_config['DEFAULT'].get('precision', 'float32'),
             watch_metric: str = run_config['DEFAULT'].get('watch_metric', 'f1_score'),
             seed: int = 1,
             milestones: list[int] = None,
             resume: str = ""
             ):
        """
        :param model: neural network used in training
        :param dataset_name: dataset used for training
        :param data_dir: data directory
        :param client_num_in_total: number of workers in a distributed cluster
        :param client_num_per_round: number of workers
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
        :param watch_metric:
        :param class_mixup:
        :param precision:
        :param analysis:
        :param seed

        Args:
            milestones:
        """
        if milestones is None:
            milestones = []
        print('Starting...')
        args = copy.deepcopy(locals())
        args.pop('self')

        device = run_config['DEFAULT']['device']
        set_seed(1)

        dataset, num_classes = load_dataset(dataset_name)
        partition, client_num_in_total, client_num_per_round = get_partition(partition_type,
                                                                             dataset_name,
                                                                             num_classes,
                                                                             client_num_in_total,
                                                                             client_num_per_round,
                                                                             alpha,
                                                                             dataset)

        run = wandb.init(
            # mode='disabled',
            project=run_config['DEFAULT']['project'],
            entity=run_config['DEFAULT']['entity'],
            name=f'{fl_algorithm}_{dataset_name}_{partition_type}_{client_num_per_round}_{client_num_in_total}_{client_optimizer}_{lr}'
                 f'_{server_optimizer}_{model}_{analysis}'
                 f'{server_lr}_{alpha}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            config=args,
        )
        wandb.config['num_samples'] = len(dataset['train'])
        client_datasets = partition(dataset['train'])
        partition_name = partition_type if partition_type != 'dirichlet' else f'{partition_type}_{alpha}'
        plot_data_distributions(dataset, dataset_name, client_datasets, num_classes)

        if 'label_noise' in analysis and dataset_name in ['wisdm_phone', 'wisdm_watch', 'widar', 'ut_har', 'casas',
                                                          'epic_sounds', 'emognition']:
            client_datasets, noise_percentages = add_label_noise(analysis, dataset_name, client_datasets, num_classes)
            plot_noise_distribution(noise_percentages)

        print('Saving dataset in object store')
        data_ref = ray.put(dataset['train'])
        print('Saving client indices in object store')
        client_dataset_refs = [ray.put(client_dataset) for client_dataset in
                               tqdm(client_datasets)]

        global_model = load_model(model_name=model, trainer=trainer, dataset_name=dataset_name)
        if resume != "" and Path(f'weights/{resume}/best_model.pt').exists():
            global_model.load_state_dict(f'weights/{resume}/best_model.pt')
        global_model = global_model.cpu()

        if trainer == 'BaseTrainer':
            from scorers.classification_evaluator import evaluate
            if dataset_name in {'energy'}:
                from scorers.regression_evaluator import evaluate
                criterion = nn.MSELoss()
                wandb.config['loss'] = 'MSE'
            elif dataset_name in {'ego4d'}:
                from scorers.localization_evaluator import evaluate
                criterion = nn.CrossEntropyLoss()
                wandb.config['loss'] = 'CrossEntropyLoss'
            else:
                criterion = nn.CrossEntropyLoss()
                wandb.config['loss'] = 'CrossEntropyLoss'
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
            # global_model.load(pt)
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                torch.optim.SGD(global_model.parameters(), lr=lr), T_0=10, T_mult=2,
                eta_min=1e-6)

            optimizer = torch.optim.SGD(global_model.parameters(),
                                        lr=lr)  # dummy optimizer meant for scheduler. do not confuse for actual optimizer
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
            if round_idx % test_frequency == 0 and round_idx > 0:
                metrics = evaluate(global_model, dataset['test'], device=device, num_classes=num_classes,
                                   batch_size=batch_size)
                v = metrics.get(watch_metric)
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                confusion_metric = None
                if 'confusion' in metrics:
                    confusion_metric = metrics['confusion'].numpy()
                    del metrics['confusion']
                if v is not None and v > best_metric:
                    best_metric = v
                    best_model = copy.deepcopy(global_model.cpu())
                    path = f'weights/{wandb.run.name}'
                    Path(path).mkdir(parents=True, exist_ok=True)
                    torch.save(best_model.state_dict(), f'{path}/best_model.pt')
                    if confusion_metric is not None:
                        chart = get_confusion_matrix_plot(confusion_metric)
                        wandb.log({'confusion_matrix': wandb.Html(chart)}, step=round_idx)
                        np.save(f'{path}/confusion_matrix.npy', confusion_metric)
                        wandb.log(
                            {'confusion_matrix_chart':
                                 wandb.Table(dataframe=pd.DataFrame(confusion_metric,
                                                                    columns=list(range(confusion_metric.shape[0]))))},
                            step=round_idx)

                wandb.log(metrics, step=round_idx)
                print(f'metric round_idx = {watch_metric}: {v}')

            local_metrics_avg, global_model, scheduler = distributed_fedavg(aggregator,
                                                                            client_trainers,
                                                                            client_dataset_refs,
                                                                            client_num_per_round,
                                                                            global_model,
                                                                            round_idx,
                                                                            scheduler,
                                                                            device,
                                                                            precision)
            print(local_metrics_avg)
            wandb.log(local_metrics_avg, step=round_idx)


if __name__ == '__main__':
    fire.Fire(Experiment)
