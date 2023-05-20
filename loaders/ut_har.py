import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class UTHarDataset(Dataset):
    def __init__(self, data: np.array, label: np.array):
        self.data = data
        self.targets = label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index, :, :, :], int(self.targets[index])


def load_dataset(root_dir='./datasets/ut_har'):
    data_list = glob.glob(root_dir + '/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir + '/UT_HAR/label/*.csv')
    ut_har_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        ut_har_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        ut_har_data[label_name] = torch.Tensor(label)
    return {
        'train': UTHarDataset(ut_har_data['X_train'], ut_har_data['y_train']),
        'val': UTHarDataset(ut_har_data['X_val'], ut_har_data['y_val']),
        'test': UTHarDataset(ut_har_data['X_test'], ut_har_data['y_val']),
    }


if __name__ == '__main__':
    dataset = load_dataset()
    print(len(dataset['train']))
    print(dataset['train'][0][0].shape)
