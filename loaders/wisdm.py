import os.path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from partition.utils import train_test_split, make_split


class WISDMDataset(Dataset):
    def __init__(self, data: Mapping[str, list[np.ndarray | int]]):
        self.data = data
        self.targets = self.data['Y']

    def __getitem__(self, index):
        try:
            return torch.tensor(self.data['X'][index], dtype=torch.float), torch.tensor(self.data['Y'][index])
        except IndexError as e:
            print("index = {}".format(index))
            print(e)
            print(len(self.data['X']))
            print(len(self.data['Y']))

    def __len__(self):
        return len(self.data['Y'])


def define_cols(df, prefix='acc'):
    columns = ['subject', 'activity', 'timestamp', f'x_{prefix}', f'y_{prefix}', f'z_{prefix}', 'null']
    df.columns = columns
    df = df.drop('null', axis=1)
    return df


def filter_merge_interval(dfa, dfg, act_df):
    t0_a = dfa['timestamp'].min()
    t0_g = dfg['timestamp'].min()
    t1_a = dfa['timestamp'].max()
    t1_g = dfg['timestamp'].max()

    t0 = max(t0_a, t0_g)
    t1 = min(t1_a, t1_g)
    dfa = dfa[(t0 <= dfa['timestamp']) & (dfa['timestamp'] <= t1)]
    dfg = dfg[(t0 <= dfg['timestamp']) & (dfg['timestamp'] <= t1)]

    df = dfa.merge(dfg.drop(dfg.columns[[0, 1]], axis=1), how='inner', on='timestamp')
    df = df.sort_values(by='timestamp')
    df = df.dropna()
    codes = act_df.code.unique()
    df = df[df.activity.isin(codes)]
    replace_codes = zip(act_df.code, act_df.fcode)
    for code, replacement_code in replace_codes:
        df['activity'] = df.activity.replace(code, replacement_code)
    return df


def process_dataset(act_df, data_path):
    dfs = []
    for i in tqdm(range(1600, 1651)):
        df_wa = define_cols(
            pd.read_csv(f'{data_path}/raw/watch/accel/data_{i}_accel_watch.txt', header=None, sep=',|;',
                        engine='python'))
        df_wg = define_cols(
            pd.read_csv(f'{data_path}/raw/watch/gyro/data_{i}_gyro_watch.txt', header=None, sep=',|;',
                        engine='python'),
            prefix='gyro')
        dfs.append(filter_merge_interval(df_wa, df_wg, act_df))
    return pd.concat(dfs)


def normalize_data(df):
    cols = [f'{axis}_{sensor}' for axis in ['x', 'y', 'z'] for sensor in ['acc', 'gyro']]
    for col in tqdm(cols):
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def get_processed_dataframe(reprocess=False):
    if os.path.exists('datasets/wisdm/processed.csv') and not reprocess:
        return pd.read_csv('datasets/wisdm/processed.csv')
    act_df = pd.read_csv('datasets/wisdm/activity_key_filtered.txt')
    processed_df = process_dataset(act_df, "datasets/wisdm/")
    processed_df = normalize_data(processed_df)
    processed_df.to_csv('datasets/wisdm/processed.csv')
    return processed_df


def create_dataset(df, clients=None, window=200, overlap=0.5):
    if clients is None:
        clients = list(range(1600, 1651))
    c_idxs = {}
    idx = 0
    X = []
    Y = []
    for client in tqdm(clients):
        c_idxs[client] = []
        data = df[df.subject == client].sort_values(by='timestamp')
        activities = data.activity.unique()
        for activity in activities:
            df_f = data[data.activity == activity]
            for i in range(window, len(df_f), int(window * overlap)):
                if i + window > len(df_f):
                    continue
                X.append(df_f[df_f.columns[3:10]].iloc[i:i + window].to_numpy())
                Y.append(activity)
                c_idxs[client].append(idx)
                idx += 1
    return {'X': X, 'Y': Y}, c_idxs


def split_dataset(data: dict, client_mapping_train: dict, client_mapping_test: dict):
    all_train, mapping_train = make_split(client_mapping_train)
    all_test, mapping_test = make_split(client_mapping_test)

    train_data = {'X': [data['X'][i] for i in all_train], 'Y': [data['Y'][i] for i in all_train]}
    test_data = {'X': [data['X'][i] for i in all_test], 'Y': [data['Y'][i] for i in all_test]}
    return WISDMDataset(train_data), WISDMDataset(test_data), {'train': mapping_train, 'test': mapping_test}



def load_dataset(window=200, overlap=0.5, reprocess=True, split=0.8):
    if os.path.exists('datasets/wisdm/wisdm.dt') and not reprocess:
        return torch.load('datasets/wisdm/wisdm.dt')
    processed_df = get_processed_dataframe(reprocess=reprocess)
    if reprocess:
        clients = list(range(1600, 1651))
        data, idx = create_dataset(processed_df, clients=clients, window=window, overlap=overlap)
        dataset = WISDMDataset(data)
        client_mapping_train, client_mapping_test = train_test_split(idx, split)
        train_dataset, test_dataset, split = split_dataset(data, client_mapping_train, client_mapping_test)

        torch.save({
            'full_dataset': dataset,
            'train': train_dataset,
            'test': test_dataset,
            'client_mapping': idx,
            'split': split
        }, 'datasets/wisdm/wisdm.dt')
    data = torch.load('datasets/wisdm/wisdm.dt')
    return data
