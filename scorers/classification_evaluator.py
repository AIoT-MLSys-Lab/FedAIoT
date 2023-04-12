import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torchmetrics import Accuracy, F1Score, Metric
from tqdm import tqdm


class LossMetric(Metric):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.add_state("loss", default=torch.tensor(0, dtype=float), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0, dtype=float), dist_reduce_fx="mean")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        # print(output.device, target.device)
        # print(self.loss, self.loss.dtype)
        l=target.size(0)*self.criterion(output, target).data.item()
        self.loss += l
        self.total += target.size(0)

    def compute(self):
        return self.loss.float() / self.total.float()


def evaluate(model, test_data, device, num_classes=12):
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

    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    metrics = {
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'f1_score': F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device),
    }
    losses = {'cross_entropy_loss': LossMetric(criterion).to(device)}
    with torch.no_grad():
        label_list, pred_list = list(), list()
        for batch_idx, (data, labels) in enumerate(tqdm(test_dataloader)):
            # for data, labels, lens in test_data:
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            output = model(data)


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
    return {k: v.compute().cpu().float() for k, v in metrics.items()} | {k: v.compute().cpu().float() for k, v in losses.items()}
