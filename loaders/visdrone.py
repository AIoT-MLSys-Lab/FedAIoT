import os.path
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from ultralytics.yolo.data import YOLODataset
from ultralytics.yolo.data.dataloaders.v5loader import LoadImagesAndLabels

from loaders.utils import ParameterDict

YOLO_HYPERPARAMETERS = {
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'fl_gamma': 0.0,
    'label_smoothing': 0.0,
    'nbs': 64,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'mask_ratio': 0.0,
    'overlap_mask': 0.0,
    'conf': 0.25,
    'iou': 0.45,
    'max_det': 1000,
    'plots': False,
    'half': False,  # use half precision (FP16)
    'dnn': False,
    'data': None,
    'imgsz': 640,
    'verbose': False
}
YOLO_HYPERPARAMETERS = ParameterDict(YOLO_HYPERPARAMETERS)
NAMES = ('pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle',
         'block', 'car_group')


class VisDroneDataset(Dataset):
    """
    A PyTorch Dataset class for the VisDrone dataset.
    """

    def __init__(self, root: str, hyp: Dict[str, Any], augment: bool = True):
        """
        Initialize the dataset.

        Args:
            root (str): Path to the root directory of the dataset.
            hyp (Dict[str, Any]): Hyperparameters dictionary.
            augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        """
        self.root = root
        self.dataset = LoadImagesAndLabels(
            path=root,
            augment=augment,
            hyp=hyp,
            # rect=True
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing the image tensor and the label.
        """
        dt = self.dataset[index]
        return dt[0].float() / 255.0, dt[1]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.dataset)


def convert_visdrone_to_yolo_format(visdrone_folder: str, output_folder: str) -> None:
    """
    Convert VisDrone dataset to YOLOv5 format.

    Args:
        visdrone_folder (str): Path to the VisDrone dataset folder.
        output_folder (str): Path to the output folder where the converted dataset will be saved.
    """
    visdrone_folder = Path(visdrone_folder)
    output_folder = Path(output_folder)

    for data_split in ["train", "val", "test"]:
        images_folder = visdrone_folder / f"{data_split}/images"
        annotations_folder = visdrone_folder / f"{data_split}/annotations"

        output_images_folder = output_folder / f"{data_split}/images"
        output_labels_folder = output_folder / f"{data_split}/labels"

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


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str, Tuple[int, int]]]) \
        -> Tuple[torch.Tensor, torch.Tensor, List[str], Tuple[Tuple[int, int], ...]]:
    """
    Custom collate function for DataLoader.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, str, Tuple[int, int]]]): List of tuples, each containing an image tensor, label tensor, image path, and a tuple of image dimensions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[str], Tuple[Tuple[int, int], ...]]: A tuple containing stacked image tensors, concatenated label tensors, list of image paths, and a tuple of image dimensions.
    """
    im, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im, 0).float(), torch.cat(label, 0), path, shapes


def load_dataset(root: str = "datasets/visdrone/yolo_format",
                 augment: bool = False,
                 hyp: Dict[str, Any] = YOLO_HYPERPARAMETERS) \
        -> Dict[str, Union[YOLODataset, Dict[str, List[int]], Dict[str, Dict[int, List[int]]]]]:
    """
    Load the VisDrone dataset with YOLO format.

    Args:
        root (str, optional): Path to the root directory of the dataset. Defaults to "datasets/visdrone/yolo_format".
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        hyp (Dict[str, Any], optional): Hyperparameters dictionary. Defaults to YOLO_HYPERPARAMETERS.

    Returns:
        Dict[str, Union[YOLODataset, Dict[str, List[int]], Dict[str, Dict[int, List[int]]]]]: A dictionary containing train, val, and test datasets, client_mapping, and split information.
    """
    dataset_train = YOLODataset(
        img_path=os.path.join(root, 'train'),
        hyp=hyp,
        augment=augment,
        names=['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle',
               'block', 'car_group']
    )

    dataset_val = YOLODataset(
        img_path=os.path.join(root, 'val'),
        hyp=hyp,
        augment=augment,
        names=['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle',
               'block', 'car_group']
    )

    dataset_test = YOLODataset(
        img_path=os.path.join(root, 'test'),
        hyp=hyp,
        augment=augment,
        names=['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle',
               'block', 'car_group']
    )

    df = pd.read_csv('split.csv', index_col='image_id')
    if not Path('visdrone_client_mapping.pt').exists():
        client_mapping = {k: [] for k in range(100)}
        for i, d in tqdm(enumerate(dataset_train)):
            p = dataset_train[i]['im_file'].split('/')[-1]
            c = df.loc[p]['cluster']
            client_mapping[c].append(i)
        torch.save(client_mapping, 'visdrone_client_mapping.pt')
    client_mapping = torch.load('visdrone_client_mapping.pt')
    return {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test,
        'client_mapping': None,
        'split': {'train': client_mapping}
    }


if __name__ == "__main__":
    # visdrone_folder = "../datasets/visdrone"
    # output_folder = "../datasets/visdrone/yolo_format"
    #
    # convert_visdrone_to_yolo_format(visdrone_folder, output_folder)
    dt = YOLODataset(
        img_path="datasets/visdrone/yolo_format/train",
        hyp=YOLO_HYPERPARAMETERS,
        names=['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle',
               'block', 'car_group']
    )
    print(dt[0].keys())
