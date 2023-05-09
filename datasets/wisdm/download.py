import os
import zipfile

import requests

# Define the URL for the dataset
WISDM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"

# Define the directory where you want to save the dataset
SAVE_DIR = "datasets/wisdm/"


# Function to download the dataset
def download_wisdm_dataset(url, save_dir='./datasets/wisdm/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename = os.path.join(save_dir, url.split("/")[-1])

    with open(filename, "wb") as f:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)

    print(f"Downloaded {filename}")

    return filename


# Function to extract the dataset
def extract_wisdm_dataset(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    print(f"Extracted dataset to {save_dir}")


# Main function to download and extract the WISDM dataset
def main():
    file_path = download_wisdm_dataset(WISDM_URL, SAVE_DIR)
    extract_wisdm_dataset(file_path, SAVE_DIR)


if __name__ == "__main__":
    main()
