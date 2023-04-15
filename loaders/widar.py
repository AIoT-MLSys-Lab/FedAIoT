import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from partition.utils import train_test_split, make_split


class WidarDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, int]]):
        """
        Initialize the WidarDataset class.

        Args:
            data (List[Tuple[np.ndarray, int]]): List of tuples containing input data and corresponding labels.
        """
        self.data = data
        self.targets = [d[1] for d in data]

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the desired data.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the input data and corresponding label.
        """
        return self.data[idx][0].reshape(22, 20, 20), self.data[idx][1]


def map_array(my_array: np.ndarray, mapping_dict: Dict[int, int]) -> np.ndarray:
    """
    Map values in a NumPy array based on a provided mapping dictionary.

    Args:
        my_array (np.ndarray): Input NumPy array to be mapped.
        mapping_dict (Dict[int, int]): Dictionary containing the mapping of input values to output values.

    Returns:
        np.ndarray: Mapped NumPy array.
    """
    mapping_func = np.vectorize(lambda x: mapping_dict.get(x, x))
    mapped_array = mapping_func(my_array)
    return mapped_array


def filter_data(datum: Tuple[np.ndarray, List[int]], selected_classes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter input data and labels based on the selected classes.

    Args:
        datum (Tuple[np.ndarray, List[int]]): Tuple containing input data and corresponding labels.
        selected_classes (List[int]): List of selected classes to filter.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing filtered input data and corresponding labels.
    """
    input_data = datum[0]
    input_labels = np.array(datum[1])
    replace_classes = {v: k for k, v in enumerate(selected_classes)}
    mask = np.isin(input_labels, selected_classes)
    filtered_array = input_data[mask, :, :]
    filtered_classes = input_labels[mask]
    filtered_classes = map_array(filtered_classes, replace_classes)

    return filtered_array, filtered_classes


def split_dataset(data: List[Tuple[np.ndarray, int]],
                  client_mapping_train: Dict[int, List[int]],
                  client_mapping_test: Dict[int, List[int]]) \
        -> Tuple[WidarDataset, WidarDataset, Dict[str, Dict[int, List[int]]]]:
    """
    Split the dataset into train and test sets based on the client mappings.

    Args:
        data (List[Tuple[np.ndarray, int]]): The input dataset as a list of tuples containing input data and corresponding labels.
        client_mapping_train (Dict[int, List[int]]): A dictionary containing the client indices for the training set.
        client_mapping_test (Dict[int, List[int]]): A dictionary containing the client indices for the test set.

    Returns:
        Tuple[WidarDataset, WidarDataset, Dict[str, Dict[int, List[int]]]]: A tuple containing the train and test WidarDatasets, and a dictionary with train and test mappings.
    """
    all_train, mapping_train = make_split(client_mapping_train)
    all_test, mapping_test = make_split(client_mapping_test)

    train_data = [data[i] for i in all_train]
    test_data = [data[i] for i in all_test]
    return WidarDataset(train_data), WidarDataset(test_data), {'train': mapping_train, 'test': mapping_test}


def load_dataset(split=[x for x in list(range(0, 17)) if x not in [0, 1, 2, 3, 15]],
                 selected_classes=[0, 3, 7, 10, 12, 14, 15, 16, 19],
                 reprocess=False):
    """
    Load and preprocess the Widar dataset.

    Args:
        split (List[int], optional): List of client indices to include in the training set. Defaults to [x for x in list(range(0, 16)) if x not in [0, 1, 2, 3, 15]].
        selected_classes (List[int], optional): List of selected classes to filter. Defaults to [0, 3, 7, 10, 12, 14, 15, 16, 19].
        reprocess (bool, optional): Whether to reprocess the dataset or use existing preprocessed data. Defaults to False.

    Returns:
        Dict[str, Union[WidarDataset, Dict[int, List[int]]]]: Dictionary containing the full_dataset, train and test datasets, client_mapping, and split information.
    """
    path = 'datasets/widar/federated'

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
