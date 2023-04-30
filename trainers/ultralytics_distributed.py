# Ultralytics YOLO 🚀, GPL-3.0 license
import logging

import ray
import torch
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.data import YOLODataset
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import Loss

from aggregators.torchcomponentrepository import TorchComponentRepository
from loaders.visdrone import YOLO_HYPERPARAMETERS
from partition.utils import IndexedSubset
from utils import WarmupScheduler


# from validator import YoloValidator

@ray.remote(num_gpus=8.0)
class DistributedUltralyticsYoloTrainer:
    def __init__(self,
                 model_path: str,
                 state_dict: dict,
                 optimizer_name,
                 lr=.1,
                 batch_size=20,
                 shuffle=True,
                 epochs=1,
                 device='cuda', amp=True,
                 args=YOLO_HYPERPARAMETERS):
        self.loss_items = None
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.args = args
        self.amp = amp
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = DetectionModel(cfg=model_path)
        self.model.args = args
        self.model.load_state_dict(state_dict)
        self.criterion = Loss(de_parallel(self.model.to(device)))
        self.loss_names = ['Loss']
        self.shuffle = shuffle
        self.optimizer = TorchComponentRepository.get_class_by_name(self.optimizer_name, torch.optim.Optimizer)(
            self.model.parameters(),
            lr=self.lr,
        )
        # self.base_scheduler = lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[30, 60])
        self.base_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # self.base_scheduler = torch.optim.lr_scheduler.MultiStepLR(torch.optim.SGD(self.model.parameters(), lr=lr),
        #                                                  milestones=[500],
        #                                                  gamma=0.1)
        self.scheduler = WarmupScheduler(self.optimizer, warmup_epochs=3, scheduler=self.base_scheduler)
        self.scaler = GradScaler(enabled=self.amp)

    def update(self, model_params, scheduler_params):
        self.model.load_state_dict(model_params)
        self.scheduler.load_state_dict(scheduler_params)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def step(self, client_idx, client_data, round_idx, device='cuda'):
        client_data = IndexedSubset(dataset=ray.get(client_data['dataset']),
                                    indices=ray.get(client_data['indices']))
        # wandb.log({'lr': self.scheduler.get_last_lr()[0]}, step=round_idx)
        losses = None
        for epoch in range(self.epochs):
            losses = self.train_one_epoch(client_idx, client_data, round_idx, device)
            self.scheduler.step()
        local_metrics = {
            'iou_loss': losses[0],
            'obj_loss': losses[1],
            'cls_loss': losses[2],
            'lr': self.scheduler.get_last_lr()[0]
        } if losses is not None else {}
        return self.get_model_params(), len(client_data), local_metrics

    def train_one_epoch(self, client_idx, client_data, round_idx, device):
        weight = len(client_data)
        client_dataloader = torch.utils.data.DataLoader(
            dataset=client_data,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=1,
            drop_last=False,
            collate_fn=YOLODataset.collate_fn
        )
        # self.model = DataParallel(self.model)
        self.model = self.model.to(device)
        self.model.train()
        print("Client ID " + str(client_idx) + " round Idx " + str(round_idx) + " Samples " + str(weight))
        logging.info(f"Client ID {client_idx} round Idx {round_idx}")
        # pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        tloss = None
        self.optimizer.zero_grad()
        print(f"Client {client_idx} Scheduler step: ", self.scheduler.get_last_lr(), "Round: ", round_idx)

        for i, batch in tqdm(enumerate(client_dataloader), total=len(client_dataloader)):
            batch = self.preprocess_batch(batch, device)
            with autocast(device_type='cuda', enabled=self.amp):
                preds = self.model(batch['img'])
                loss, loss_items = self.criterion(preds, batch)
            tloss = (tloss * i + loss_items) / (i + 1) if tloss is not None \
                else loss_items

            # Backward
            self.scaler.scale(loss).backward()
            self.optimizer_step()
            torch.cuda.empty_cache()
        return tloss.cpu().numpy()

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def preprocess_batch(self, batch, device):
        batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
        return batch

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(self.model)
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    # def plot_metrics(self):
    #     pass

    # def set_model_attributes(self):
    #     self.model.nc = 12  # attach number of classes to model
    #     self.model.names = self.data['names']  # attach class names to model
    #     self.model.args = self.args  # attach hyperparameters to model
    #     # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    # def plot_training_samples(self, batch, ni):
    #     plot_images(images=batch['img'],
    #                 batch_idx=batch['batch_idx'],
    #                 cls=batch['cls'].squeeze(-1),
    #                 bboxes=batch['bboxes'],
    #                 paths=batch['im_file'],
    #                 fname=self.save_dir / f'train_batch{ni}.jpg')

    # def plot_metrics(self):
    #     plot_results(file=self.csv)  # save results.png
    #
    # def plot_training_labels(self):
    #     boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
    #     cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
    #     plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)
