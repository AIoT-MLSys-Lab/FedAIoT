import glob

import numpy as np
import torch


def process(root_dir='.'):
    data_list = glob.glob(root_dir + '/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir + '/UT_HAR/label/*.csv')
    print(data_list, label_list)
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


if __name__ == '__main__':
    data = process()
    for k, v in data.items():
        print(k, v.shape)
