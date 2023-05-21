from collections.abc import Iterable
from pathlib import Path
from typing import List, Sized

import altair as alt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset
        self.targets = [dataset.targets[i] for i in indices]

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
            class_counts[int(datasets[i].targets[j])] += 1
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
    Path('logs/').mkdir(exist_ok=True)
    (heatmap + text).save('logs/class_dist.html'), data_bar.save('logs/data_dist.html')
    return 'logs/class_dist.html', 'logs/data_dist.html'
