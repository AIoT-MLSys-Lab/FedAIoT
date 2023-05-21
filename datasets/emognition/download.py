import os
import zipfile

import gdown

# Define the shared Google Drive file URL
FILE_ID = "1XUYMBP0p2VSJTppgE2BT87rr6MK0vagl"

# Define the directory where you want to save the dataset
SAVE_DIR = "./datasets/emognition"


# Function to download the file from Google Drive
def download_file_from_google_drive(file_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "_study_data.zip")
    gdown.download(output=file_path, quiet=False, id=file_id)

    return file_path


# Function to extract the dataset
def extract_file(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)
    print(f"Extracted dataset to {save_dir}")


# Main function to download and extract the WidarData.zip file
def main():
    file_path = download_file_from_google_drive(FILE_ID, SAVE_DIR)
    extract_file(file_path, SAVE_DIR)


if __name__ == "__main__":
    main()
