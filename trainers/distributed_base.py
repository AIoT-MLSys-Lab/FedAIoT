import configparser
import logging
import os
import warnings

import numpy as np
import ray
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

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


@ray.remote(num_gpus=1.0/num_trainers_per_gpu)
class DistributedTrainer:
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

            # momentum=0.9,
            # weight_decay=5e-4,
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
        # client_data = IndexedSubset(dataset=ray.get(client_data['dataset']),
        #                             indices=ray.get(client_data['indices']))
        if len(client_data) < self.batch_size:
            self.batch_size = len(client_data)
        weight = len(client_data)
        client_dataloader = DataLoader(
            dataset=client_data,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=4,
            drop_last=False,
        )
        print(torch.cuda.is_initialized())
        # print(torch.cudnn.version())
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())

        self.model.to(device)
        self.model.train()
        print("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " Samples " + str(weight))
        if len(client_dataloader) < 1:
            warnings.warn("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " data_loader " + str(len(
                client_dataloader)))
        logging.info(f"Client ID {client_idx} round Idx {round_idx}")

        criterion = self.criterion.to(device)
        optimizer = self.optimizer

        epoch_loss = []
        print(f"Client {client_idx} Scheduler step: ", self.scheduler.get_last_lr(), "Round: ", round_idx)
        for epoch in range(self.epochs):
            batch_loss = []
            loss = np.nan
            for batch_idx, (data, labels) in tqdm(enumerate(client_dataloader), total=len(client_dataloader)):
                if len(labels) <= 1:
                    continue
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(device_type=device, ):  # Enable mixed precision training
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
                self.scaler.scale(loss).backward()

                # Uncomment the line below if you want to use gradient clipping
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                # Replace optimizer.step() with the following line to update the weights and scale the gradients
                self.scaler.step(optimizer)

                # Update the scaler
                self.scaler.update()

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
