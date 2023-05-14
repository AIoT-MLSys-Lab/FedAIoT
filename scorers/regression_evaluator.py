import torch
from sklearn.metrics import r2_score
from torch import nn
from torchmetrics import MeanAbsoluteError, R2Score
from torchmetrics import Metric
from tqdm import tqdm

from scorers.utils import LossMetric


# class R2Score(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.preds = []
#         self.targets = []
#
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         self.preds += preds.reshape((-1,)).cpu().tolist()
#         self.targets += target.reshape((-1,)).cpu().tolist()
#
#     def compute(self):
#         return torch.tensor(r2_score(self.targets, self.preds))


def evaluate(model, test_data, device):
    model.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        shuffle=False,
        batch_size=32,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )
    model.eval()

    criterion = nn.MSELoss().to(device)
    metrics = {
        'mae': MeanAbsoluteError().to(device),
        'R^2': R2Score().to(device)
    }
    losses = {'L2 Loss': LossMetric(criterion).to(device)}
    with torch.no_grad():
        label_list, pred_list = list(), list()
        for batch_idx, (data, labels) in enumerate(tqdm(test_dataloader)):
            # for data, labels, lens in test_data:
            # labels = labels.type(torch.float)
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            labels = labels.reshape((-1,))
            output = output.reshape((-1,))
            for lm in losses.values():
                lm.update(output, labels)
            # pred = output.data.max(1, keepdim=True)[1]
            for mm in metrics.values():
                mm.update(output, labels)
            # pred = output.data.max(1, keepdim=True)[
            #     1
            # ]  # get the index of the max log-probability
            # correct = pred.eq(labels.data.view_as(pred)).sum()
            # for idx in range(len(labels)):
            #     label_list.append(labels.detach().cpu().numpy()[idx])
            #     pred_list.append(pred.detach().cpu().numpy()[idx][0])
            #
            # metrics["test_correct"] += correct.item()
            # metrics["test_loss"] += loss * labels.size(0)
            # metrics["test_total"] += labels.size(0)
    return {k: v.compute().cpu().float() for k, v in metrics.items()} | {k: v.compute().cpu().float() for k, v in
                                                                         losses.items()}
