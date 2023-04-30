from collections.abc import Iterable
from typing import List, Sized

import altair as alt
import numpy as np
import pandas as pd
import ray
import torch
from torch.utils.data import Dataset


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset

    def __getitem__(self, index):
        try:
            i = self.indices[index]
            dt = self.dataset[i]
        except KeyError or IndexError:
            print(type(self))
            print("index = {}".format(index))
            print("i = {}".format(i))
            print(type(self.indices))
            print(self.indices)
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def train_test_split(client_mapping, split):
    if type(split) is float and split <= 1.0:
        train_clients = np.random.choice(list(client_mapping.keys()), int(len(client_mapping.keys()) * split),
                                         replace=False)
    elif isinstance(split, Iterable) and all(isinstance(item, int) for item in split):
        train_clients = list(split)
    elif type(split) is int and split < len(client_mapping.keys()):
        train_clients = np.random.choice(list(client_mapping.keys()), split, replace=False)
    else:
        raise ValueError("Invalid split value: {}".format(split))
    test_clients = list(set(client_mapping.keys()) - set(train_clients))
    return {k: client_mapping[k] for k in train_clients}, {k: client_mapping[k] for k in test_clients}


def make_split(client_mapping_train):
    indices = []
    mapping_train = {k: [] for k in client_mapping_train.keys()}
    i = 0
    for k, v in client_mapping_train.items():
        indices += v
        for _ in range(len(v)):
            mapping_train[k].append(i)
            i += 1
    return indices, mapping_train


def compute_client_data_distribution(datasets: List[Sized | Dataset], num_classes: int):
    class_distribution = []
    data_distribution = []

    for i in range(len(datasets)):
        class_counts = torch.zeros(num_classes)
        for j in range(len(datasets[i])):
            class_counts[datasets[i][j][1]] += 1
        class_counts = class_counts.numpy()
        data_distribution.append(np.sum(class_counts))
        class_counts = class_counts / np.sum(class_counts)
        class_distribution.append(class_counts)
    return data_distribution, class_distribution


def get_html_plots(data_distribution, class_distribution):
    data = []
    num_clients = len(data_distribution)
    for i in range(len(class_distribution[0])):
        for j in range(len(class_distribution)):
            data.append({"client": j, "class": i, "value": class_distribution[j][i]})

    heatmap = (
        alt.Chart(pd.DataFrame(data))
        .mark_rect()
        .encode(
            x=alt.X("client:N", title="Client"),
            y=alt.Y("class:N", title="Class"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="yellowgreenblue"),
                            legend=alt.Legend(title="Percentage of Samples")),
            tooltip="value:Q",
        )
        .properties(
            title=alt.TitleParams(
                "Class Distribution of Clients",
                fontSize=12,
            ),
            # width=200,
            # height=120,
        )
    )

    text = (
        alt.Chart(pd.DataFrame(data))
        .mark_text()
        .encode(
            x=alt.X("client:N"),
            y=alt.Y("class:N"),
            text=alt.Text("value:Q", format=".2f", ),
            color=alt.condition(
                alt.datum.value > 0.5, alt.value("black"), alt.value("white")
            ),
        )
        .transform_filter((alt.datum.value > 0.01))
    )

    data_bar = (
        alt.Chart(pd.DataFrame({"client": range(num_clients), "value": data_distribution}))
        .mark_bar()
        .encode(
            x=alt.X("client:N", title="Client", axis=alt.Axis(labelFontSize=8)),
            y=alt.Y("value:Q", title="Data Samples", axis=alt.Axis(labelFontSize=8)),
            tooltip="value:Q",
        )
        .properties(
            title=alt.TitleParams(
                "Sample Distribution of Clients",
                fontSize=12,
            ),
            # width=200,
            # height=120,
        )
    )
    (heatmap + text).save('logs/class_dist.html'), data_bar.save('logs/data_dist.html')
    return 'logs/class_dist.html', 'logs/data_dist.html'

def label_nosiy(client_datasets, class_num, error_ratio, error_var):
    """
    Add label noise to client datasets.

    Args:
        client_datasets: a list of client datasets
        class_num: an integer indicating the number of classes.
        error_ratio: a float between 0 and 1 indicating the ratio of labels to be flipped.
        error_var: a float indicating the variance of the Gaussian distribution used to determine
            the level of label noise.

    Returns:
        A list of client datasets
    """
    client_datasets_label_error = []
    for original_data in client_datasets:
        # Determine the level of label noise for this client dataset. The level is computed by normal distribution
        noisy_level = np.random.normal(error_ratio, error_var)
        if noisy_level < 0:
            noisy_level = 0

        # Set the level of sparsity in the noise matrix.
        sparse_level = 0.4

        # Create a probability matrix for each label, where each element represents the probability of a label being assigned to that image.
        prob_matrix = [1-noisy_level] * class_num * class_num

        # Set a random subset of elements in the probability matrix to zero to create sparsity.
        sparse_elements = np.random.choice(class_num*class_num, round(class_num*(class_num-1)*sparse_level))
        for idx in range(len(sparse_elements)):
            # Ensure that the diagonal elements of the probability matrix are not set to zero.
            while sparse_elements[idx]%(class_num+1) == 0:
                sparse_elements[idx] = np.random.choice(class_num*class_num, 1)
            prob_matrix[sparse_elements[idx]] = 0
        
        available_spots = np.argwhere(np.array(prob_matrix) == 1 - noisy_level)
        for idx in range(class_num):
            available_spots = np.delete(available_spots, np.argwhere(available_spots == idx*(class_num+1)))
        for idx in range(class_num):
            row = prob_matrix[idx*4:(idx*4)+4]
            if len(np.where(np.array(row) == 1 - noisy_level)[0]) == 2:
                unsafe_points = np.where(np.array(row) == 1 - noisy_level)[0]
                unsafe_points = np.delete(unsafe_points, np.where(np.array(unsafe_points) == idx*(class_num+1))[0])
                available_spots = np.delete(available_spots, np.argwhere(available_spots == unsafe_points[0]))
            if np.sum(row) == 1 - noisy_level:
                zero_spots = np.where(np.array(row) == 0)[0]
                prob_matrix[zero_spots[0] + idx * 4], prob_matrix[available_spots[0]] = prob_matrix[available_spots[0]], prob_matrix[zero_spots[0] + idx * 4]
                available_spots = np.delete(available_spots, 0) 

        prob_matrix = np.reshape(prob_matrix, (class_num, class_num))

        for idx in range(len(prob_matrix)):
            zeros = np.count_nonzero(prob_matrix[idx]==0)
            if class_num-zeros-1 == 0:
                prob_element = 0
            else:
                prob_element = (noisy_level) / (class_num-zeros-1)
            prob_matrix[idx] = np.where(prob_matrix[idx] == 1-noisy_level, prob_element, prob_matrix[idx])
            prob_matrix[idx][idx] = 1-noisy_level
        
        tmp_dataset = []
        for i in range(len(original_data)):
            tmp_dataset_cell = [0, 0]
            # add label nosiy
            orginal_label = original_data[i][1].numpy()
            new_label = np.random.choice(class_num,p=prob_matrix[orginal_label])
            tmp_dataset_cell.append(new_label)
            original_raw_data = original_data[i][0].numpy()
            tmp_dataset_cell.append(original_raw_data)
            tmp_dataset.append(tmp_dataset_cell)
        client_datasets_label_error.append(tmp_dataset)
    return client_datasets_label_error
        