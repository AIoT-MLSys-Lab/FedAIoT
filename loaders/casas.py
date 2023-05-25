#!/usr/bin/env python3

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_dataset(datasetName='all'):
    X = np.load('./datasets/casas/npy/' + datasetName + '-x.npy')
    Y = np.load('./datasets/casas/npy/' + datasetName + '-y.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    X_tensor = torch.from_numpy(X.astype(int))
    y_tensor = torch.from_numpy(Y.astype(int))

    X_tensor_train = torch.from_numpy(X_train.astype(int))
    y_tensor_train = torch.from_numpy(Y_train.astype(int))

    X_tensor_test = torch.from_numpy(X_test.astype(int))
    y_tensor_test = torch.from_numpy(Y_test.astype(int))
    # Create a PyTorch Dataset using TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
    test_dataset = TensorDataset(X_tensor_test, y_tensor_test)
    dataset.targets = y_tensor
    train_dataset.targets = y_tensor_train
    test_dataset.targets = y_tensor_test
    data_dict = {
        'full_dataset': dataset,
        'train': train_dataset,
        'test': test_dataset
    }
    return data_dict


if __name__ == '__main__':
    dt = load_dataset()
    print(len(dt['train']))
    print(dt['train'][0][0].shape)