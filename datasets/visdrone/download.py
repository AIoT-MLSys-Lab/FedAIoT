import os
import shutil
import zipfile
from pathlib import Path

import PIL.Image as Image
import gdown
import requests
from tqdm import tqdm

# Define the VisDrone dataset URLs
DATASET_URLS = [
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-train.zip",
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-val.zip",
    "https://downloads.visdrone.org/data2018/VisDrone2018-DET-test-challenge.zip"
]

FILE_IDs = [
    ('1i8iZ-zYBgWwzX9355HIYrWM1uKeqWW0S', 'VisDrone2019-DET-train.zip', 'train'),
    ('1qJKZdv2jEv2c7SfEdMwWR3KOyj_mfhBN', 'VisDrone2019-DET-val.zip', 'val'),
    ('1nTC4cqNqT_IJ7EIH28i9YTVGNFq5WgqL', 'VisDrone2019-DET-test-dev.zip', 'test')
]

FOLODER_SPLITS = [
    ('VisDrone2019-DET-train', 'train'),
    ('VisDrone2019-DET-val', 'val'),
    ('VisDrone2018-DET-test-dev', 'test')
]

# Define the directory where you want to save the dataset
SAVE_DIR = "./datasets/visdrone"


def convert_visdrone_to_yolo_format() -> None:
    """
    Convert VisDrone dataset to YOLOv5 format.
    """
    visdrone_folder = Path(SAVE_DIR)

    for folder, split in FOLODER_SPLITS:
        images_folder = visdrone_folder / f"{folder}/images"
        annotations_folder = visdrone_folder / f"{folder}/annotations"

        output_images_folder = visdrone_folder / f"{split}/images"
        output_labels_folder = visdrone_folder / f"{split}/labels"

        output_images_folder.mkdir(parents=True, exist_ok=True)
        output_labels_folder.mkdir(parents=True, exist_ok=True)

        for annotation_file in tqdm(annotations_folder.glob("*.txt")):
            image_file = images_folder / f"{annotation_file.stem}.jpg"

            if image_file.exists():
                # Copy image file
                shutil.copy(image_file, output_images_folder / image_file.name)
                img = Image.open(image_file).convert("RGB")
                # Convert and save label file
                with open(annotation_file) as f:
                    lines = f.readlines()

                with open(output_labels_folder / annotation_file.name, "w") as f:
                    for line in lines:
                        items = line.strip().split(",")

                        # Calculate normalized values required by YOLOv5
                        # class_id, x_center, y_center, width, height

                        class_id = int(items[5])
                        x_center = (int(items[0]) + int(items[2]) / 2) / img.width
                        y_center = (int(items[1]) + int(items[3]) / 2) / img.height
                        width = int(items[2]) / img.width
                        height = int(items[3]) / img.height

                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


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


def download_file_from_google_drive(file_id, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, filename)
    gdown.download(output=file_path, quiet=False, id=file_id)
    return file_path


# Function to extract the dataset
def extract_dataset(file_path, save_dir):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    print(f"Extracted dataset to {save_dir}")


# Main function to download and extract the VisDrone dataset
def main():
    s_dir = SAVE_DIR
    for file_id, filename, split in FILE_IDs:
        file_path = download_file_from_google_drive(file_id=file_id,
                                                    save_dir=SAVE_DIR,
                                                    filename=filename)
        if 'test' in file_path:
            s_dir = f'{SAVE_DIR}/VisDrone2018-DET-test-dev'
            Path(s_dir).mkdir(exist_ok=True)
        extract_dataset(file_path, s_dir)
        print(file_path)
    convert_visdrone_to_yolo_format()


if __name__ == "__main__":
    main()
