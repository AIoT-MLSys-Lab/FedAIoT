import configparser
import logging
import warnings

import numpy as np
import ray
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aggregators.torchcomponentrepository import TorchComponentRepository
from trainers.distributed_base import BaseTrainer, create_dataloader, mixup_data, mixup_criterion
from utils import set_seed, read_system_variable

system_config = configparser.ConfigParser()
system_config.read('system.yml')
num_gpus, num_trainers_per_gpu, seed = read_system_variable(system_config)
print(f'Seed is {seed}')
set_seed(seed)


@ray.remote(num_gpus=1.0 / num_trainers_per_gpu, )
class ScaffoldTrainer(BaseTrainer):
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
                         amp)
        self.control_variates = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

    def step(self, client_idx, client_data, round_idx, device='cuda', global_control_variates=None):

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

                output = self.model(data)
                loss = criterion(output, labels)

                if torch.isnan(loss):
                    print('nan loss observed')
                    continue

                loss.backward()

                # Apply Scaffold control variates if provided
                if global_control_variates:
                    for param, c_diff in zip(self.model.parameters(), global_control_variates.values()):
                        param.grad += c_diff.to(device)

                optimizer.step()

                batch_loss.append(loss.item())

                torch.cuda.empty_cache()

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.scheduler.step()

        # Calculate local control variates
        local_control_variates = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        if global_control_variates:
            for param, control_var in zip(self.model.named_parameters(), global_control_variates.values()):
                name, param = param
                param, control_var = param.to('cpu'), control_var.to('cpu')
                local_control_variates[name] = param.data - control_var.data

        local_update_state = self.model.cpu().state_dict()
        local_metrics = {'Local Loss': np.mean(epoch_loss) if epoch_loss != [] else 0,
                         'learning_rate': self.scheduler.get_last_lr()[0]}

        return local_update_state, local_control_variates, len(client_data), local_metrics

    def step_low_precision(self, client_idx, client_data, round_idx, precision='float32', device='cuda',
                           global_control_variates=None):
        batch_size = self.batch_size
        if len(client_data) < self.batch_size:
            batch_size = len(client_data)
        weight = len(client_data)
        client_dataloader = create_dataloader(dataset=client_data,
                                              batch_size=batch_size,
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
            warnings.warn("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " data_loader " + str(
                len(client_dataloader)))
        logging.info(f"Client ID {client_idx} round Idx {round_idx}")

        criterion = self.criterion.to(device)

        optimizer = TorchComponentRepository.get_class_by_name(self.optimizer_name, torch.optim.Optimizer)(
            self.model.parameters(),
            lr=self.lr,
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
                    if labels.dtype == torch.float32:
                        labels = labels.half()
                elif precision == 'float64':
                    data = data.to(device).double()
                    if labels.dtype == torch.float32:
                        labels = labels.double()
                    labels = labels.to(device)
                else:
                    data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                if self.mixup != 1:
                    data, labels_a, labels_b, lam = mixup_data(data, labels, alpha=self.mixup)
                output = self.model(data)
                if self.dataset_name == 'energy':
                    output = output.reshape((-1,))
                if self.mixup != 1:
                    loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
                else:
                    loss = criterion(output, labels)

                loss.backward()
                torch.cuda.empty_cache()

                if global_control_variates:
                    for param, c_diff in zip(self.model.parameters(), global_control_variates.values()):
                        param.grad += c_diff.to(device)

                optimizer.step()
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

        local_control_variates = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        if global_control_variates:
            for param, control_var in zip(self.model.parameters(), global_control_variates.values()):
                local_control_variates[param] = param.data - control_var.data

        local_update_state = self.model.cpu().state_dict()
        local_metrics = {'Local Loss': sum(epoch_loss) / len(epoch_loss),
                         'learning_rate': self.scheduler.get_last_lr()[0]}

        return local_update_state, local_control_variates, weight, local_metrics
