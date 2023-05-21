import os
from typing import List

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False,
                                           download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return {'train': trainset, 'test': testset, 'label_names': classes}


def compute_client_data_distribution(datasets: List[Dataset], num_classes: int):
    class_distribution = []
    data_distribution = []

    for i in range(len(datasets)):
        class_counts = torch.zeros(num_classes)
        for j in range(len(datasets[i].targets)):
            class_counts[datasets[i].targets[j]] += 1
        class_counts = class_counts.numpy()
        data_distribution.append(np.sum(class_counts))
        class_counts = class_counts / np.sum(class_counts)
        class_distribution.append(class_counts)
    return data_distribution, class_distribution


def visualize_client_data_distribution(datasets: List[Dataset], num_clients: int, num_classes: int):
    data_distribution, class_distribution = compute_client_data_distribution(datasets, num_classes)

    # create a heatmap of the data distribution for each client
    fig, ax = plt.subplots()
    im = ax.imshow(np.array(class_distribution).T, cmap='YlGn')

    # add text annotations for each cell
    for i in range(len(class_distribution[0])):
        for j in range(len(class_distribution)):
            text = ax.text(j, i, class_distribution[j][i], ha="center", va="center", color="black")

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set tick labels and axis labels
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ax.set_xticks(np.arange(len(class_distribution)))
    ax.set_yticks(np.arange(len(class_distribution[0])))
    ax.set_xticklabels([f"{i}" if i % 10 == 0 else '' for i in range(len(class_distribution))])
    ax.set_yticklabels([f"{i}" for i in range(len(class_distribution[0]))])
    ax.set_xlabel("Client")
    ax.set_ylabel("Class")
    ax.set_title("Class Distribution of Clients")

    plt.show()

    fig, ax = plt.subplots()
    ax.bar(range(num_clients), data_distribution)
    ax.set_xlabel("Client")
    ax.set_ylabel("Data Samples")
    ax.set_title("Sample Distribution of Clients")
    plt.show()
    plt.savefig("sample_distribution_matplotlib.png")


def vis_data_distribution_altair(data_distribution, class_distribution):
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

    return alt.vconcat(heatmap + text, data_bar)

