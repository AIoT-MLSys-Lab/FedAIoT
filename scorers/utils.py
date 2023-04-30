import torch
from torchmetrics import Metric


class LossMetric(Metric):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.add_state("loss", default=torch.tensor(0, dtype=float), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0, dtype=float), dist_reduce_fx="mean")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        # print(output.device, target.device)
        # print(self.loss, self.loss.dtype)
        l = target.size(0) * self.criterion(output, target).data.item()
        self.loss += l
        self.total += target.size(0)

    def compute(self):
        return self.loss.float() / self.total.float()
