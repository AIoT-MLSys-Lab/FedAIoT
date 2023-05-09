import glob
import os
import zipfile
from pathlib import Path

import gdown
import numpy as np
import torch
from tqdm import tqdm

# Define the shared Google Drive file URL
FILE_ID = "14vp4D8W0X2bDLpXnpP-U_VT9PIGkVf_4"

# Define the directory where you want to save the dataset
SAVE_DIR = "./datasets/widar"


# Function to download the file from Google Drive
def download_file_from_google_drive(file_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "Widardata.zip")
    gdown.download(output=file_path, quiet=False, id=file_id)

    return file_path


# Function to extract the dataset
def extract_file(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Extracted dataset to {save_dir}")


def process():
    files = glob.glob('./datasets/widar/Widardata/*/*/*.csv')
    data = {}
    for file in tqdm(files):
        y = int(file.split('/')[-2].split('-')[0])
        user = int(file.split('/')[-1].split('-')[0].replace('user', ''))
        if user not in data.keys():
            data[user] = {'X': [], 'Y': []}
        x = np.genfromtxt(file, delimiter=',')
        data[user]['X'].append(x)
        data[user]['Y'].append(y)
    Path('./datasets/widar/federated').mkdir(exist_ok=True)
    for user in data.keys():
        X = np.concatenate(np.expand_dims(np.array(data[user]['X']), 0))
        Y = np.array(data[user]['Y'])
        print(f'{user}_data.pkl')
        print(X.shape, Y.shape)
        torch.save((X, Y), f'./datasets/widar/federated/{user}.pkl')


# Main function to download and extract the WidarData.zip file
def main():
    file_path = download_file_from_google_drive(FILE_ID, SAVE_DIR)
    extract_file(file_path, SAVE_DIR)
    process()


if __name__ == "__main__":
    main()
