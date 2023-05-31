import configparser
import logging
import os
import warnings

import numpy as np
import ray
import torch
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from aggregators.torchcomponentrepository import TorchComponentRepository
from models.utils import load_model
from utils import WarmupScheduler

system_config = configparser.ConfigParser()
system_config.read('system.yml')
num_gpus = int(os.environ['num_gpus']) if 'num_gpus' in os.environ else system_config['DEFAULT'].getint('num_gpus', 1)
num_trainers_per_gpu = int(os.environ['num_trainers_per_gpu']) if 'num_gpus' in os.environ else system_config[
    'DEFAULT'].getint(
    'num_trainers_per_gpu', 1)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    # Get random permutation for batch
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Mixup data
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Create label/mixup label pairs
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, weights=None):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    if weights is not None:
        loss_a = (loss_a * weights).sum(1).mean()
        loss_b = (loss_b * weights).sum(1).mean()
    return lam * loss_a + (1 - lam) * loss_b


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


def create_dataloader(dataset, batch_size=64, shuffle=None, pin_memory=None, num_workers=None):
    if False and hasattr(dataset, 'targets'):  # If the dataset has a 'targets' attribute
        # Count the number of samples per class
        class_counts = {}
        for label in dataset.targets:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Calculate weights
        weights = [1.0 / (1 + class_counts[label]) for label in dataset.targets]

        # Create sampler and dataloader
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                drop_last=False,
                                )
    else:
        # If the dataset does not have a 'targets' attribute, use a regular DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )

    return dataloader


class BaseTrainer:
    def __init__(self, model_name: str,
                 dataset_name: str,
                 state_dict: dict,
                 criterion,
                 optimizer_name,
                 lr,
                 batch_size,
                 class_mixup,
                 shuffle=True,
                 scheduler='LinearLR', gamma=1, milestones=[],
                 epochs=1,
                 amp=True, ):
        set_seed(1)
        from aggregators.torchcomponentrepository import TorchComponentRepository
        self.dataset_name = dataset_name
        self.scheduler = None
        self.amp = amp
        self.model = load_model(model_name=model_name, trainer='BaseTrainer', dataset_name=dataset_name)
        self.model.load_state_dict(state_dict)
        self.criterion = criterion
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.optimizer = None
        self.param_size = None if self.model is None else sum(p.numel() for p in self.model.parameters())
        self.lr = lr
        self.mixup = class_mixup
        self.shuffle = shuffle
        self.pin_memory = True
        self.num_workers = 1
        self.epochs = epochs
        self.schedule = list(range(75, 300, 75))
        self.optimizer = TorchComponentRepository.get_class_by_name(self.optimizer_name, torch.optim.Optimizer)(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        # print(TorchRepo.name2cls("linearlr", torch.optim.lr_scheduler.LRScheduler))
        self.base_scheduler = TorchComponentRepository.get_class_by_name(scheduler,
                                                                         torch.optim.lr_scheduler.LRScheduler)(
            self.optimizer,
            gamma=gamma,
            milestones=milestones)
        self.scheduler = WarmupScheduler(self.optimizer, warmup_epochs=0, scheduler=self.base_scheduler)
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[75, 125])
        self.scaler = GradScaler(enabled=self.amp)

    def update(self, model_params, scheduler_params):
        self.model.load_state_dict(model_params)
        self.scheduler.load_state_dict(scheduler_params)

    # def set_model(self, model, args):
    #     self.model = model
    #     self.args = args
    #     self.param_size = sum(p.numel() for p in self.model.parameters())

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def step(self, client_idx, client_data, round_idx, device='cuda'):

        if len(client_data) < self.batch_size:
            batch_size = len(client_data)
        else:
            batch_size = self.batch_size

        client_dataloader = create_dataloader(dataset=client_data, batch_size=batch_size, shuffle=self.shuffle,
                                              pin_memory=self.pin_memory, num_workers=4)

        self.model.to(device)
        self.model.train()

        criterion = self.criterion.to(device)
        optimizer = self.optimizer

        epoch_loss = []

        for epoch in range(self.epochs):
            batch_loss = []

            for batch_idx, (data, labels) in tqdm(enumerate(client_dataloader)):

                if len(labels) < 1:
                    continue

                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(device_type=device):  # Enable mixed precision training
                    output = self.model(data)

                    if self.dataset_name == 'energy':
                        output = output.reshape((-1,))

                    loss = criterion(output, labels)

                if torch.isnan(loss):
                    print('nan loss observed')
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                batch_loss.append(loss.item())

                torch.cuda.empty_cache()

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.scheduler.step()
        # if not epoch_loss:
        #     print(f'length of dataloader:{epoch_loss}, {len(client_data)}, {len(client_dataloader)}')
        #     raise ValueError(
        #         f'length of dataloader:{epoch_loss},{batch_loss}, {len(client_data)}, {len(client_dataloader)}')
        local_update_state = self.model.cpu().state_dict()
        local_metrics = {'Local Loss': np.mean(epoch_loss) if epoch_loss != [] else 0,
                         'learning_rate': self.scheduler.get_last_lr()[0]}

        return local_update_state, len(client_data), local_metrics

    def step_low_precision(self, client_idx, client_data, round_idx, precision='float32', device='cuda'):
        # client_data = IndexedSubset(dataset=ray.get(client_data['dataset']),
        #                             indices=ray.get(client_data['indices']))
        if len(client_data) < self.batch_size:
            self.batch_size = len(client_data)
        weight = len(client_data)
        client_dataloader = create_dataloader(dataset=client_data,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              pin_memory=self.pin_memory,
                                              num_workers=4)
        model = self.model
        model.to(device)
        model.train()

        if precision == 'float16':
            model.half()  # convert to half precision
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
        if precision == 'float64':
            model.double()  # convert to double precision   

        print("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " Samples " + str(weight))
        if len(client_dataloader) < 1:
            warnings.warn("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " data_loader " + str(len(
                client_dataloader)))
        logging.info(f"Client ID {client_idx} round Idx {round_idx}")

        criterion = self.criterion.to(device)

        optimizer = TorchComponentRepository.get_class_by_name(self.optimizer_name, torch.optim.Optimizer)(
            self.model.parameters(),
            lr=self.lr,

            # momentum=0.9,
            # weight_decay=5e-4,
        )

        epoch_loss = []
        print(f"Client {client_idx} Scheduler step: ", self.scheduler.get_last_lr(), "Round: ", round_idx)
        for epoch in range(self.epochs):
            batch_loss = []
            loss = np.nan
            for batch_idx, (data, labels) in tqdm(enumerate(client_dataloader), total=len(client_dataloader)):
                if len(labels) <= 1:
                    continue
                if precision == 'float16':
                    data = data.to(device).half()
                    labels = labels.to(device)
                elif precision == 'float64':
                    data = data.to(device).double()
                    labels = labels.to(device)
                else:
                    data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                # with autocast(device_type=device, dtype=torch.float16):  # Enable mixed precision training
                if self.mixup != 1:
                    data, labels_a, labels_b, lam = mixup_data(data, labels, alpha=self.mixup)
                output = self.model(data)
                if self.dataset_name == 'energy':
                    output = output.reshape((-1,))
                if self.mixup != 1:
                    loss = mixup_criterion(
                        criterion,
                        output,
                        labels_a,
                        labels_b,
                        lam
                    )
                else:
                    loss = criterion(output, labels)

                # Replace loss.backward() with the following lines to scale the loss and update the gradients
                loss.backward()
                torch.cuda.empty_cache()
                # Uncomment the line below if you want to use gradient clipping
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                # Replace optimizer.step() with the following line to update the weights and scale the gradients
                optimizer.step()

                # Update the scaler
                # self.scaler.update()

                batch_loss.append(loss.item())

            print(f'Client Index = {client_idx}\tEpoch: {epoch}\tBatch Loss: {loss:.6f}\tBatch Number: {batch_idx}')
            logging.info(
                f"Client Index = {client_idx}\tEpoch: {epoch}\tBatch Loss: {loss:.6f}\tBatch Number: {batch_idx}")
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                warnings.warn("Batch loss is empty")
                epoch_loss.append(np.nan)
        self.scheduler.step()
        local_update_state = self.model.cpu().state_dict()
        local_metrics = {'Local Loss': sum(epoch_loss) / len(epoch_loss), 'learning_rate':
            self.scheduler.get_last_lr()[0]}
        return local_update_state, weight, local_metrics


@ray.remote(num_gpus=1.0 / num_trainers_per_gpu)
class DistributedTrainer(BaseTrainer):
    def __init__(self, model_name: str,
                 dataset_name: str,
                 state_dict: dict,
                 criterion,
                 optimizer_name,
                 lr,
                 batch_size,
                 class_mixup,
                 shuffle=True,
                 scheduler='LinearLR', gamma=1, milestones=[],
                 epochs=1,
                 amp=True, ):
        super().__init__(model_name,
                         dataset_name,
                         state_dict,
                         criterion,
                         optimizer_name,
                         lr,
                         batch_size,
                         class_mixup,
                         shuffle,
                         scheduler, gamma, milestones,
                         epochs,
                         amp, )
