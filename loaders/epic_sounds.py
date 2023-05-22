import os
import time
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score
from tqdm import tqdm

from loaders.pack_audio import pack_audio
from loaders.spec_augment import combined_transforms
from loaders.utils import pack_pathway_output


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


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 1000
    return sec


class EpicSoundsRecord(object):
    def __init__(self, tup, sampling_rate=24000):
        self._index = str(tup[0])
        self._series = tup[1]
        self.sampling_rate = sampling_rate

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def video_id(self):
        return self._series['video_id']

    @property
    def annotation_id(self):
        return self._series['annotation_id']

    @property
    def start_audio_sample(self):
        return int(timestamp_to_sec(self._series["start_timestamp"]) * self.sampling_rate)

    @property
    def end_audio_sample(self):
        return int(timestamp_to_sec(self._series["stop_timestamp"]) * self.sampling_rate)

    @property
    def label(self):
        return self._series["class_id"] if "class_id" in self._series else 0

    @property
    def num_audio_samples(self):
        return self.end_audio_sample - self.start_audio_sample


class Epicsounds(torch.utils.data.Dataset):

    def __init__(self, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC Sounds".format(mode)
        self.mode = mode

        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = 5

        # self.audio_dataset = pickle.load(open(cfg.EPICSOUNDS.AUDIO_DATA_FILE, 'rb'))
        self.audio_dataset = None
        print("Constructing EPIC Sounds {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        ANNOTATIONS_DIR = "datasets/epic_sounds/epic-sounds-annotations-main"
        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(ANNOTATIONS_DIR, "EPIC_Sounds_train.pkl")]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(ANNOTATIONS_DIR, "EPIC_Sounds_validation.pkl")]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(ANNOTATIONS_DIR, "EPIC_Sounds_validation.pkl")]
        else:
            path_annotations_pickle = [os.path.join(ANNOTATIONS_DIR, file)
                                       for file in [ANNOTATIONS_DIR, "EPIC_Sounds_validation.pkl"]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(file)

        self._video_records = []
        self._temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._video_records.append(
                        EpicSoundsRecord(tup, 24000)
                    )
                    self._temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load Audio Annotations split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        print(
            "Constructing audio annotations dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )
        self.targets = [x.label for x in self._video_records]

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.audio_dataset is None:
            self.audio_dataset = h5py.File("datasets/epic_sounds/EPIC_audio.hdf5", 'r')

        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]

        spectrogram = pack_audio(
            self.audio_dataset,
            self._video_records[index],
            temporal_sample_index
        )
        # Normalization.
        spectrogram = spectrogram.float()
        if self.mode in ["train", "train+val"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = combined_transforms(spectrogram)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self.targets[index]
        # spectrogram = pack_pathway_output(spectrogram)

        # metadata = {
        #     "annotation_id": self._video_records[index].annotation_id
        # }

        return spectrogram, label

    def __len__(self):
        return len(self._video_records)


def load_dataset():
    return {
        'train': Epicsounds('train'),
        'test': Epicsounds('test'),
    }

# if __name__ == '__main__':
#     model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=44).cuda()
#     train_data = Epicsounds('train')
#     test_data = Epicsounds('test')
#     train_loader = torch.utils.data.DataLoader(
#         train_data,
#         batch_size=64,
#         shuffle=True,
#         num_workers=1
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=64,
#         shuffle=True,
#         num_workers=10
#     )
#     # Define the loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.001)
#
#     for epoch in range(10):  # loop over the dataset multiple times
#         running_loss = 0.0
#         for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
#             # Get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#
#             # Transfer to GPU
#
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.8)
#             inputs, labels_a, labels_b = inputs.cuda(), labels_a.cuda(),  labels_b.cuda()
#             outputs = model(inputs)
#             # loss = criterion(outputs, labels)
#             loss = mixup_criterion(
#                 criterion,
#                 outputs,
#                 labels_a,
#                 labels_b,
#                 lam
#             )
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.cpu().item()
#             if i % 10 == 1:  # print every 2000 mini-batches
#                 print(labels.shape)
#                 print(outputs.shape)
#                 print(inputs.shape)
#                 print(max(labels))
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#
#     print('Finished Training')
#     # Testing
#     correct = 0
#     total = 0
#     model.eval()  # switch model to evaluation mode
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels, _, _ = data
#             images, labels = images.cuda(), labels.cuda()
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on test images: %d %%' % (
#             100 * correct / total))
