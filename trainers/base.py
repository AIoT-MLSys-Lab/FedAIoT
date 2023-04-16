import logging

import torch
import wandb

from aggregators.torchcomponentrepository import TorchComponentRepository


class BaseTrainer:
    def __init__(self, model_path: str,
                 state_dict: dict,
                 criterion,
                 optimizer_name,
                 lr,
                 batch_size=20,
                 shuffle=True,
                 epochs=1):
        self.scheduler = None
        from torchvision.models import resnet
        from torch.optim import lr_scheduler
        from models.wisdm import LSTM_NET
        from models.widar import Widar_ResNet18
        LSTM_NET
        self.model = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.criterion = criterion
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.optimizer = None
        self.param_size = None if self.model is None else sum(p.numel() for p in self.model.parameters())
        self.lr = lr
        self.batch_size = 20
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
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[75, 125])

    def update(self, model_params):
        self.model.load_state_dict(model_params)

    # def set_model(self, model, args):
    #     self.model = model
    #     self.args = args
    #     self.param_size = sum(p.numel() for p in self.model.parameters())

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def step(self, client_idx, client_data, round_idx, device='cuda'):
        weight = len(client_data)
        client_dataloader = torch.utils.data.DataLoader(
            dataset=client_data,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=1,
            drop_last=False,
        )
        self.model.to(device)
        self.model.train()
        print("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " Samples " + str(weight))
        logging.info(f"Client ID {client_idx} round Idx {round_idx}")

        criterion = self.criterion.to(device)
        optimizer = self.optimizer

        epoch_loss = []
        print(f"Client {client_idx} Scheduler step: ", self.scheduler.get_last_lr(), "Round: ", round_idx)
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(client_dataloader):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            print(f'Client Index = {client_idx}\tEpoch: {epoch}\tBatch Loss: {loss:.6f}\tBatch Number: {batch_idx}')
            logging.info(
                f"Client Index = {client_idx}\tEpoch: {epoch}\tBatch Loss: {loss:.6f}\tBatch Number: {batch_idx}")
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.scheduler.step()
        wandb.log({"Local Loss": sum(epoch_loss) / len(epoch_loss), 'learning_rate': self.scheduler.get_last_lr()[0]},  step=round_idx)
        local_update_state = self.model.cpu().state_dict()
        return local_update_state, weight
