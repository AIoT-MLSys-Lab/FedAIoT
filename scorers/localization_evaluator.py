import numpy as np
import torch
from torch import nn
from torchmetrics import Accuracy, Metric
from tqdm import tqdm

from scorers.utils import LossMetric


def keyframe_distance(preds, uid_list):
    distance_list = list()
    sec_list = list()
    for pred, gt in zip(preds, uid_list):
        clip_length = gt['json_parent_end_sec'].item() - gt['json_parent_start_sec'].item()
        clip_frames = gt['json_parent_end_frame'].item() - gt['json_parent_start_frame'].item() + 1
        fps = clip_frames / clip_length
        keyframe_loc_pred = np.argmax(pred)
        keyframe_loc_pred = np.argmax(pred)
        keyframe_loc_pred_mapped = (gt['json_parent_end_frame'].item() - gt[
            'json_parent_start_frame'].item()) / 16 * keyframe_loc_pred
        keyframe_loc_gt = gt['pnr_frame'].item() - gt['json_parent_start_frame'].item()
        err_frame = abs(keyframe_loc_pred_mapped - keyframe_loc_gt)
        err_sec = err_frame / fps
        distance_list.append(err_frame.item())
        sec_list.append(err_sec.item())
    # When there is no false positive
    if len(distance_list) == 0:
        # Should we return something else here?
        return 0, 0
    return np.array(distance_list), np.array(sec_list)


class KeyframeDistance(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("distance_list", default=[], dist_reduce_fx="cat")
        self.add_state("sec_list", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, infos: list[torch.Tensor]):
        distance_list = list()
        sec_list = list()
        preds = preds.cpu().numpy()
        preds = preds[:, :-1]
        # pnr_frames = infos['pnr_frame']
        # clip_start_secs = infos['clip_start_sec']
        # clip_end_secs = infos['clip_end_sec']
        # clip_start_frames = infos['clip_start_frame']
        # clip_end_frames = infos['clip_end_frame']
        for pred, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, pnr_frame in zip(preds,
                                                                                                   *infos):
            # print(clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, pnr_frame)
            if pnr_frame.item() == -1:
                continue
            clip_length = clip_start_sec.item() - clip_end_sec.item()
            clip_frames = clip_end_frame.item() - clip_start_frame.item() + 1
            fps = clip_frames / clip_length
            keyframe_loc_pred = np.argmax(pred)
            keyframe_loc_pred_mapped = (clip_end_frame.item() - clip_start_frame.item()) / 16 * keyframe_loc_pred
            keyframe_loc_gt = pnr_frame.item() - clip_start_frame.item()
            err_frame = abs(keyframe_loc_pred_mapped - keyframe_loc_gt)
            err_sec = err_frame / fps
            distance_list.append(err_frame.item())
            sec_list.append(err_sec.item())
        # When there is no false positive
        if len(distance_list) == 0:
            # Should we return something else here?
            return
        self.sec_list.extend(sec_list)
        self.distance_list.extend(distance_list)

    def compute(self):
        # Perform any final computations here.
        # This might just be converting your lists of distances and seconds to tensors.
        # Make sure to handle the case where the lists are empty.
        return torch.mean(torch.tensor(self.distance_list))


def evaluate(model, test_data, device, num_classes=12, batch_size=32):
    model.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    metrics = {
        'avg_multilabel_accuracy': Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device),
        # 'binary_accuracy': Accuracy(task="multiclass", num_classes=1).to(device),
        # 'f1_score': F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device),
        'keyframe_dist': KeyframeDistance().to(device),
    }
    lbl_type = torch.LongTensor
    losses = {
        'cce_loss': LossMetric(criterion).to(device),
    }

    with torch.no_grad():
        label_list, pred_list = list(), list()
        for batch_idx, (data, labels, info) in enumerate(tqdm(test_dataloader)):
            # for data, labels, lens in test_data:
            labels = labels.type(lbl_type)
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            for lm in losses.values():
                lm.update(output, labels)
            # pred = output.data.max(1, keepdim=True)[1]

            for name, mm in metrics.items():
                if name == 'keyframe_dist':
                    mm.update(output, info)
                    continue
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
