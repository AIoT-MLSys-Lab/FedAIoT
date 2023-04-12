import os
import zipfile

import requests

# Define the VisDrone dataset URLs
DATASET_URLS = [
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-train.zip",
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-val.zip",
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-test-challenge.zip"
]

# Define the directory where you want to save the dataset
SAVE_DIR = "visdrone_dataset"


# Function to download the dataset
def download_dataset(url, save_dir):
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
def extract_dataset(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    print(f"Extracted dataset to {save_dir}")


# Main function to download and extract the VisDrone dataset
def main():
    for url in DATASET_URLS:
        file_path = download_dataset(url, SAVE_DIR)
        extract_dataset(file_path, SAVE_DIR)


if __name__ == "__main__":
    main()
