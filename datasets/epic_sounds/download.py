import os
import zipfile

import gdown

# Define the shared Google Drive file URL
FILE_ID = "1BAaBIYqU6gZDyFqu9aW6spvpwpsEDZMS"

# Define the directory where you want to save the dataset
SAVE_DIR = "./datasets/epic_sounds"


# Function to download the file from Google Drive
def download_file_from_google_drive(file_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "EPIC_audio.hdf5")
    gdown.download(output=file_path, quiet=False, id=file_id)

    return file_path


# Main function to download and extract the WidarData.zip file
def main():
    download_file_from_google_drive(FILE_ID, SAVE_DIR)


if __name__ == "__main__":
    main()
