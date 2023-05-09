import torch
from torch import nn
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

from scorers.utils import LossMetric


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

    criterion = nn.L1Loss().to(device)
    metrics = {
        'mae': MeanAbsoluteError().to(device),
    }
    losses = {'L1 Loss': LossMetric(criterion).to(device)}
    with torch.no_grad():
        label_list, pred_list = list(), list()
        for batch_idx, (data, labels) in enumerate(tqdm(test_dataloader)):
            # for data, labels, lens in test_data:
            # labels = labels.type(torch.float)
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
    return {k: v.compute().cpu().float() for k, v in metrics.items()} | {k: v.compute().cpu().float() for k, v in
                                                                         losses.items()}
