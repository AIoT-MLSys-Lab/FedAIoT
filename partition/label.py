import numpy as np
import torch

from partition.utils import IndexedSubset


class DisjointLabelPartition:
    def __init__(self, num_users, num_classes=10, max_class_per_user=2):
        self.num_users = num_users
        self.num_classes = num_classes
        self.max_unique_class_per_user = max_class_per_user
        self.label_split = None

    def __call__(self, dataset):
        class_indices_dict = {i: [] for i in range(self.num_classes)}
        client_data_indices_dict = {i: [] for i in range(self.num_users)}
        label = np.array(dataset.targets)
        for i in range(len(label)):
            label_i = label[i].item()
            class_indices_dict[label_i].append(i)

        num_classes = self.num_classes
        shard_per_user = self.max_unique_class_per_user
        label_idx_split = class_indices_dict

        shard_per_class = int(shard_per_user * self.num_users / num_classes)

        for label_i in label_idx_split:
            label_idx = label_idx_split[label_i]
            num_leftover = len(label_idx) % shard_per_class
            leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
            new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
            new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()

            for i, leftover_label_idx in enumerate(leftover):
                new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])

            label_idx_split[label_i] = new_label_idx

        if self.label_split is None:
            label_split = list(range(num_classes)) * shard_per_class
            label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
            label_split = np.array(label_split).reshape((self.num_users, -1)).tolist()

            for i in range(len(label_split)):
                label_split[i] = np.unique(label_split[i]).tolist()

            self.label_split = label_split

        for i in range(self.num_users):
            for label_i in self.label_split[i]:
                idx = torch.arange(len(label_idx_split[label_i]))[
                    torch.randperm(len(label_idx_split[label_i]))[0]].item()
                client_data_indices_dict[i].extend(label_idx_split[label_i].pop(idx))
        dataset_ref = dataset
        return [IndexedSubset(dataset_ref, v) for _, v in client_data_indices_dict.items()]
