import json
from pathlib import Path
from typing import Dict, List, Tuple, Mapping

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from tqdm import tqdm

import partition.utils as p_utils


# from partition.utils import train_test_split, make_split


class EmognitionDataset(Dataset):
    """
    A PyTorch Dataset class for the WISDM dataset.
    """

    def __init__(self, data: Mapping[str, list[np.ndarray | int]]):
        """
        Initialize the dataset with data mapping.
        Args:
            data (Mapping[str, list[np.ndarray | int]]): A dictionary containing the data and targets.
        """
        self.data = data
        self.targets = self.data['Y']

    def __getitem__(self, index):
        """
        Get an item from the dataset by index.
        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and target tensors for the specified index.
        """
        return torch.tensor(self.data['X'][index], dtype=torch.float), \
            torch.tensor(self.data['Y'][index])

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data['Y'])


SIGNAL_FREQUENCY = {
    'BVP': 64,
    'GSR': 4,
    'TEMP': 4,
    'ACC': 32,
    'ACC_X': 32,
    'ACC_Y': 32,
    'ACC_Z': 32,
}


def get_muse_sampling():
    return 256


def filter_data(data, fs, lowcut, highcut, filter_order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


fs = 4  # Sample frequency (replace with your actual value)


def normalize_time_series(time_series):
    # Compute the min and max values for each feature across all samples and time points
    min_values = np.min(time_series, axis=(0, 1), keepdims=True)
    max_values = np.max(time_series, axis=(0, 1), keepdims=True)

    # Normalize each feature using min-max normalization
    normalized_time_series = (time_series - min_values) / (max_values - min_values)

    return normalized_time_series


DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S:%f'

MUSE_DTYPES = {
    'TimeStamp': 'datetime64[ms]', 'Delta_TP9': float, 'Delta_AF7': float,
    'Delta_AF8': float, 'Delta_TP10': float, 'Theta_TP9': float,
    'Theta_AF7': float, 'Theta_AF8': float, 'Theta_TP10': float,
    'Alpha_TP9': float, 'Alpha_AF7': float, 'Alpha_AF8': float, 'Alpha_TP10': float,
    'Beta_TP9': float, 'Beta_AF7': float, 'Beta_AF8': float, 'Beta_TP10': float,
    'Gamma_TP9': float, 'Gamma_AF7': float, 'Gamma_AF8': float, 'Gamma_TP10': float,
    'RAW_TP9': float, 'RAW_AF7': float, 'RAW_AF8': float, 'RAW_TP10': float,
    'AUX_RIGHT': float, 'Accelerometer_X': float, 'Accelerometer_Y': float,
    'Accelerometer_Z': float, 'Gyro_X': float, 'Gyro_Y': float, 'Gyro_Z': float,
    'HeadBandOn': float, 'HSI_TP9': float, 'HSI_AF7': float, 'HSI_AF8': float,
    'HSI_TP10': float, 'Battery': float, 'Elements': 'string',
}

PARTICIPANTS_ID = list(range(22, 65))

DEVICES = ('EMPATICA', 'SAMSUNG_WATCH', 'MUSE')

MOVIE_TYPES = ('NEUTRAL', 'DISGUST', 'ANGER', 'AMUSEMENT', 'SURPRISE', 'AWE',
               'ENTHUSIASM', 'LIKING', 'BASELINE', 'SADNESS', 'FEAR')

MOVIE_CODES = {k: v for v, k in enumerate(MOVIE_TYPES)}

MOVIES_WITH_LABELS = [
    ('AMUSEMENT', 'AM'),
    ('ANGER', 'AN'),
    ('AWE', 'AW'),
    ('DISGUST', 'D'),
    ('ENTHUSIASM', 'E'),
    ('FEAR', 'F'),
    ('LIKING', 'L'),
    ('SADNESS', 'SA'),
    ('SURPRISE', 'SU'),
    ('BASELINE', 'B'),
    ('NEUTRAL', 'N')
]
EMOTIONS = [M for M, _ in MOVIES_WITH_LABELS if M not in {'BASELINE', 'NEUTRAL'}]
EMOTIONS_LABELS = [L for M, L in MOVIES_WITH_LABELS if M not in {'BASELINE', 'NEUTRAL'}]
MOVIES = [M for M, _ in MOVIES_WITH_LABELS]
MOVIES_LABELS = [L for _, L in MOVIES_WITH_LABELS]

STIMULATION_TIMES = {
    'AMUSEMENT': pd.Timedelta('00:02:00'),
    'ANGER': pd.Timedelta('00:02:00'),
    'AWE': pd.Timedelta('00:01:56'),
    # 'BASELINE': pd.Timedelta('00:05:00'),
    'DISGUST': pd.Timedelta('00:01:08'),
    'ENTHUSIASM': pd.Timedelta('00:01:59'),
    'FEAR': pd.Timedelta('00:02:00'),
    'LIKING': pd.Timedelta('00:01:51'),
    'NEUTRAL': pd.Timedelta('00:02:01'),
    'SADNESS': pd.Timedelta('00:01:59'),
    'SURPRISE': pd.Timedelta('00:00:49'),
    'WASHOUT': pd.Timedelta('00:02:00')
}

EMOTIONS_CATEGORICAL = ('DISGUST', 'ANGER', 'AMUSEMENT', 'SURPRISE', 'AWE',
                        'ENTHUSIASM', 'LIKING', 'SADNESS', 'FEAR')
EMOTIONS_SAM = ('VALENCE', 'AROUSAL', 'MOTIVATION')

STUDY_PHASES = ('WASHOUT', 'STIMULUS', 'QUESTIONNAIRES')
phase_code = {k: v for v, k in enumerate(STUDY_PHASES)}
BASELINE_PHASES = ('STIMULUS', 'QUESTIONNAIRES')


# read json file with signal or questionnaire data
def load_json_file(path):
    if path[-5:] != '.json':
        path += '.json'

    with open(path) as json_file:
        json_data = json.load(json_file)

    return json_data


dataset_path = 'datasets/emognition'


def split_dataset(data: Tuple[np.ndarray, int],
                  client_mapping_train: Dict[int, List[int]],
                  client_mapping_test: Dict[int, List[int]]):
    all_train, mapping_train = p_utils.make_split(client_mapping_train)
    all_test, mapping_test = p_utils.make_split(client_mapping_test)

    train_data = {'X': data[0][all_train, :, :], 'Y': data[1][all_train]}
    test_data = {'X': data[0][all_test, :, :], 'Y': data[1][all_test]}
    return EmognitionDataset(train_data), EmognitionDataset(test_data), {'train': mapping_train, 'test': mapping_test}


def resample_data(timestamps, values, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs) + 1
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, values, kind='linear', fill_value='extrapolate')
    new_values = interpolator(new_timestamps)
    return new_timestamps, new_values


def load_muse_data_for_participants(participants=PARTICIPANTS_ID, ):
    all_questionnaire_files = [f'datasets/emognition/{participant_id}/{participant_id}_QUESTIONNAIRES.json'
                               for participant_id in participants]
    genders_ages, movies_ratings, emotions_ratings, film_seen_before_study, drugs = get_questionnaire_data(
        all_questionnaire_files
    )
    device = 'MUSE'
    signals = ['Delta_AF7', 'Delta_AF8', 'Theta_AF7', 'Theta_AF8', 'Alpha_AF7', 'Alpha_AF8', 'Beta_AF7', 'Beta_AF8',
               'Gamma_AF7', 'Gamma_AF8']
    data_dict = {k: {'data': [], 'targets': []} for k in participants}
    for p_id in participants:
        p_data = load_json_file(f'{dataset_path}/{p_id}/{p_id}_QUESTIONNAIRES.json')
        for movie in p_data['metadata']['movie_order']:
            study_phase = 'STIMULUS'
            try:
                phase_data = load_json_file(f'{dataset_path}/{p_id}/{p_id}_{movie}_{study_phase}_{device}.json')
                df = pd.DataFrame(phase_data)
                timestamps = pd.to_datetime(df.TimeStamp, format='%Y-%m-%dT%H:%M:%S.%f')
                df = df[signals]
                df.index = timestamps
            except FileNotFoundError:
                print(f'There is no file:\n'
                      f'{p_id}_{movie}_{study_phase}_{device}.json')
                print('Check data_completness.csv file for information about missing files\n')
                continue
            df = df.fillna(-1)
            data_dict[p_id]['data'].append(df)
            data_dict[p_id]['targets'].append(movies_ratings[movie].loc[p_id].to_numpy()[9] / 9.0)
    return data_dict


def process_data(dct, window_length=int(10 / .10), overlap=0.8):
    data = []
    targets = []
    client_mapping = {}
    for p_id in dct.keys():
        for dt, mv in zip(dct[p_id]['data'], dct[p_id]['targets']):
            if len(dt) < window_length:
                continue
            if p_id not in client_mapping.keys():
                client_mapping[p_id] = []
            dt = dt.to_numpy()
            for i in range(0, len(dt), int(window_length * (1 - overlap))):
                if i + window_length > len(dt):
                    break
                data.append(np.expand_dims(dt[i:i + window_length, :], 0))
                targets.append(mv)
                client_mapping[p_id].append(len(data) - 1)
    data = np.concatenate(data, axis=0)
    targets = np.array(targets)
    median_targets = np.median(targets, axis=0)
    targets = np.where(targets > median_targets, 1, 0)
    return data, targets, client_mapping


def load_empatica_data_for_participants(participants=PARTICIPANTS_ID, ):
    all_questionnaire_files = [f'datasets/emognition/{participant_id}/{participant_id}_QUESTIONNAIRES.json'
                               for participant_id in participants]
    genders_ages, movies_ratings, emotions_ratings, film_seen_before_study, drugs = get_questionnaire_data(
        all_questionnaire_files
    )
    device = 'EMPATICA'
    signals = ['BVP', 'TEMP', 'IBI', 'ACC', 'EDA']
    data_dict = {k: {'data': [], 'targets': []} for k in participants}
    for p_id in participants:
        p_data = load_json_file(f'{dataset_path}/{p_id}/{p_id}_QUESTIONNAIRES.json')
        s_dt = {k: None for k in signals}

        for movie in p_data['metadata']['movie_order']:
            phases = BASELINE_PHASES if movie == 'BASELINE' else STUDY_PHASES
            s_dt_l = []
            for signal in signals:
                signal_data, timestamps, movie_labels, phase_labels = [], [], [], []
                for study_phase in phases:
                    # if study_phase != 'STIMULUS':
                    #     continue
                    try:
                        phase_data = load_json_file(f'{dataset_path}/{p_id}/{p_id}_{movie}_{study_phase}_{device}.json')
                        signal_data += [i[1] for i in phase_data[signal]]
                        timestamps += [i[0] for i in phase_data[signal]]
                        phase_labels += [phase_code[study_phase]] * len(phase_data[signal])
                    except FileNotFoundError:
                        print(f'There is no file:\n'
                              f'{p_id}_{movie}_{study_phase}_{device}.json')
                        print('Check data_completness.csv file for information about missing files\n')
                s_dt[signal] = pd.DataFrame(index=pd.to_datetime(timestamps, format="%Y-%m-%dT%H:%M:%S:%f"), data={
                    signal: signal_data})
                s_dt['phase'] = pd.DataFrame(index=pd.to_datetime(timestamps, format="%Y-%m-%dT%H:%M:%S:%f"), data={
                    'phase': phase_labels})
                if signal in ['IBI', 'BVP', 'EDA', 'TEMP']:
                    s_dt_l.append(s_dt[signal])
                    if signal == 'BVP':
                        s_dt_l.append(s_dt['phase'])
            df = pd.concat(s_dt_l, axis=1)
            df = df.resample('10ms').mean()
            df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            df.IBI = filter_data(df.IBI, 100, 0.01, 0.4, 4)
            df.EDA = filter_data(df.EDA, 100, 0.05, 2.0, 4)  # Change the highcut to 2.0 to avoid the error
            df.BVP = filter_data(df.BVP, 100, 0.5, 5, 4)
            df = df.resample('250ms').mean()
            df = df.fillna(-1)
            data_dict[p_id]['data'].append(df)
            data_dict[p_id]['targets'].append(movies_ratings[movie].loc[p_id].to_numpy()[9] / 9.0)
    return data_dict


def load_bracelet_data(split=0.8, window_length=40, overlap=0.25, participants=PARTICIPANTS_ID, reprocess=False):
    if Path('datasets/emognition/bracelet_data.dt').exists() and not reprocess:
        return torch.load('datasets/emognition/bracelet_data.dt')
    dct = load_empatica_data_for_participants(participants=participants)
    d, t, cl_idx = process_data(dct, window_length=window_length, overlap=overlap)
    d = normalize_time_series(d)
    dataset = EmognitionDataset({'X': d, 'Y': t})
    client_mapping_train, client_mapping_test = p_utils.train_test_split(cl_idx, split)
    train_dataset, test_dataset, split = split_dataset((d, t), client_mapping_train, client_mapping_test)
    torch.save({
        'full_dataset': dataset,
        'train': train_dataset,
        'test': test_dataset,
        'client_mapping': cl_idx,
        'split': split
    }, 'datasets/emognition/bracelet_data.dt')
    return torch.load('datasets/emognition/bracelet_data.dt')


def get_questionnaire_data(all_questionnaire_files):
    genders_ages, drugs = [], []
    movies_ratings = {M: [] for M in MOVIES}
    emotions_ratings = {E: {M: dict() for M in MOVIES} for E in EMOTIONS + list(EMOTIONS_SAM)}
    film_seen_before_study = pd.DataFrame(0, index=[0], columns=sorted(EMOTIONS + ['NEUTRAL']))

    for f_path in tqdm(all_questionnaire_files):
        with open(f_path, 'r') as f:
            f_dict = json.load(f)
        part_id = f_dict['metadata']['id']
        drugs.append(f_dict['metadata']['other_drugs_in_last_8h'])

        # genders and ages list
        genders_ages.append(
            pd.DataFrame(
                data={
                    'age': f_dict['metadata']['age'],
                    'gender': f_dict['metadata']['gender']
                },
                index=[part_id]
            )
        )

        # films seen before study
        seen_films_data = {k.upper(): v for k, v in f_dict['metadata']['movies_seen_before_study'].items()}
        film_seen_before_study += pd.DataFrame(seen_films_data, [0])

        # questionnaires list
        questionnaires = f_dict['questionnaires']

        # for each questionnaire append dataframe containing answers
        # to appropriate list in movies_ratings dict
        for q in questionnaires:
            # within conditions
            data = pd.DataFrame(
                data={**q['emotions'], **q['sam']},
                index=[part_id]
            )
            movies_ratings[q['movie']].append(data)

            # between conditions
            for emotion, rating in {**q['emotions'], **q['sam']}.items():
                emotions_ratings[emotion][q['movie']].setdefault(part_id, rating)

    genders_ages = pd.concat(genders_ages, ignore_index=False)
    movies_ratings = {m: pd.concat(l, ignore_index=False) for m, l in movies_ratings.items()}
    emotions_ratings = {e: pd.DataFrame(m) for e, m in emotions_ratings.items()}
    return genders_ages, movies_ratings, emotions_ratings, film_seen_before_study, drugs


def load_muse_data(split=0.8, window_length=256 * 10, overlap=0.8, reprocess=False):
    if Path('datasets/emognition/muse_data.dt').exists() and not reprocess:
        return torch.load('datasets/emognition/muse_data.dt')
    dct = load_muse_data_for_participants()
    d, t, cl_idx = process_data(dct, window_length=window_length, overlap=overlap)
    d = normalize_time_series(d)
    dataset = EmognitionDataset({'X': d, 'Y': t})
    client_mapping_train, client_mapping_test = p_utils.train_test_split(cl_idx, split)
    train_dataset, test_dataset, split = split_dataset((d, t), client_mapping_train, client_mapping_test)
    torch.save({
        'full_dataset': dataset,
        'train': train_dataset,
        'test': test_dataset,
        'client_mapping': cl_idx,
        'split': split
    }, 'datasets/emognition/muse_data.dt', pickle_protocol=5)
    return torch.load('datasets/emognition/muse_data.dt')


if __name__ == '__main__':
    dt = load_bracelet_data(window_length=40, reprocess=True, participants=[22, 23])
    # print(len(dt['full_dataset']))
    print(len(dt['train']))
    # print(len(dt['test']))
    # print(dt['client_mapping'].keys())
    # print(dt['split']['train'].keys())
    # print(dt['split']['test'].keys())
