import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from partition.utils import train_test_split, make_split


class WidarDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [d[1] for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0].reshape(22, 20, 20), self.data[idx][1]


def map_array(my_array, mapping_dict):
    # Use np.vectorize() to create a function that maps values to replacements
    mapping_func = np.vectorize(lambda x: mapping_dict.get(x, x))

    # Use the mapping function to replace values in the NumPy array
    mapped_array = mapping_func(my_array)

    return mapped_array


def filter_data(datum, selected_classes):
    input_data = datum[0]
    input_labels = np.array(datum[1])
    replace_classes = {v: k for k, v in enumerate(selected_classes)}
    mask = np.isin(input_labels, selected_classes)
    filtered_array = input_data[mask, :, :]
    filtered_classes = input_labels[mask]
    filtered_classes = map_array(filtered_classes, replace_classes)

    return filtered_array, filtered_classes


# def load_dataset(split=[x for x in list(range(0, 16)) if x not in [0, 1, 2, 3, 15]],
#                  test_clients=[0, 1, 2, 3, 15],
#                  reprocess=False):
#     path = 'datasets/widar/'
#
#     dt = os.listdir(path)
#     dtt = []
#     for i in dt:
#         if i.endswith('.pkl'):
#             try:
#                 with open(f'{path}/{i}', 'rb') as f:
#                     dtt.append(torch.load(f))
#             except pickle.UnpicklingError as e:
#                 print(i)
#     dt = dtt
#     # dt = [torch.load(f'{path}/{i}') for i in dt]
#     selected_classes = [0, 3, 7, 10, 12, 14, 15, 16, 19]
#     # dt = [[(d[0], replace_classes[d[1]]) for d in dtt if d[-1] in classes] for dtt in dt]
#     dt.sort(key=lambda x: len(x[-1]))
#     dt = [filter_data(d, selected_classes) for d in dt]
#     for d in dt:
#         print(len(d[0]), len(d[1]))
#
#     cl_idx = {}
#     i = 0
#     print(split)
#     for j in split:
#         d = dt[j]
#         cl_idx[i] = list(range(i, i + len(d[0])))
#         i += len(d[0])
#
#     train_dataset = WidarDataset([dt[i] for i in split])
#     test_dataset = WidarDataset([dt[i] for i in test_clients])
#     return {'train': train_dataset, 'test': test_dataset, 'client_mapping': cl_idx}


def split_dataset(data: list[tuple], client_mapping_train: dict, client_mapping_test: dict):
    all_train, mapping_train = make_split(client_mapping_train)
    all_test, mapping_test = make_split(client_mapping_test)

    train_data = [data[i] for i in all_train]
    test_data = [data[i] for i in all_test]
    return WidarDataset(train_data), WidarDataset(test_data), {'train': mapping_train, 'test': mapping_test}


def load_dataset(split=[x for x in list(range(0, 16)) if x not in [0, 1, 2, 3, 15]],
                 selected_classes=[0, 3, 7, 10, 12, 14, 15, 16, 19],
                 reprocess=False):
    path = 'datasets/widar/'

    data = os.listdir(path)
    dtt = []
    for i in data:
        if i.endswith('.pkl'):
            try:
                with open(f'{path}/{i}', 'rb') as f:
                    dtt.append(torch.load(f))
            except pickle.UnpicklingError as e:
                print(f'Error loading {i}')
    data = dtt
    data.sort(key=lambda x: len(x[-1]))
    data = [filter_data(d, selected_classes) for d in data]
    all_users = list(range(0, len(data)))
    cl_idx = {}
    i = 0
    for j in all_users:
        d = data[j]
        cl_idx[j] = list(range(i, i + len(d[0])))
        i += len(d[0])

    x = [d[0] for d in data]
    x = np.concatenate(x, axis=0, dtype=np.float32)
    x = (x - .0025) / .0119
    y = np.concatenate([d[1] for d in data])
    data = [(x[i], y[i]) for i in range(len(x))]
    dataset = WidarDataset(data)
    data = [dataset[i] for i in range(len(dataset))]
    client_mapping_train, client_mapping_test = train_test_split(cl_idx, split)
    train_dataset, test_dataset, split = split_dataset(data, client_mapping_train, client_mapping_test)
    data_dict = {
        'full_dataset': dataset,
        'train': train_dataset,
        'test': test_dataset,
        'client_mapping': cl_idx,
        'split': split
    }
    return data_dict
