import os

import numpy as np
import pandas as pd
import torch
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pycocotools import coco
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = models.resnet50(pretrained=True)


def extract_imagenet_features(img_path, transform=transformations, model=model):
    # Load the pre-trained ResNet50 model

    # Set the model to evaluation mode
    model.eval()

    # Define the image pre-processing transforms

    # Load the image and apply the pre-processing transforms
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)

    # Extract features from the image using the model
    with torch.no_grad():
        features = model(img_tensor)

    # Flatten the features tensor
    flattened_features = features.flatten()

    return flattened_features


class VisDroneDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # self.coco = coco.COCO(ann_dir)
        self.data_dir = data_dir
        self.images = [x for x in os.listdir(data_dir) if '.jpg' in x]
        self.transform = transform
        # self.images = os.listdir(os.path.join(data_dir, 'images'))
        # self.annotations = os.listdir(os.path.join(data_dir, 'annotations'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        # image = Image.open(os.path.join(self.data_dir, self.coco.loadImgs(self.images[idx])[0]['file_name']))
        image = extract_imagenet_features(os.path.join(self.data_dir, self.images[idx]))
        if self.transform:
            image = self.transform(image)

        # Load the annotations
        # with open(os.path.join(self.data_dir, 'annotations', self.annotations[idx]), 'r') as f:
        #     annotations = f.readlines()

        # Parse the annotations
        # boxes = []
        # labels = []
        # for annotation in annotations:
        #     xmin, ymin, xmax, ymax, label = annotation.strip().split(',')
        #     boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        #     labels.append(int(label))

        # Convert the annotations to tensors
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, self.images[idx]  # boxes, labels


# Define the transformations to be applied to the images
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = VisDroneDataset(data_dir='train/images',
                          transform=None)

# Load a pretrained ResNet-50 model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Extract features for each image in the dataset
ids = []
features = []
for i, (image_features, img_id) in tqdm(enumerate(dataset), total=len(dataset)):
    with torch.no_grad():
        feature = image_features.numpy()
    ids.append(img_id)
    features.append(feature)

# Convert the features to a numpy array
features = np.array(features)

# Perform K-means clustering on the features to cluster the images into 100 clusters
from sklearn.cluster import KMeans, DBSCAN

kmeans = KMeans(n_clusters=10).fit(features, )
df = pd.DataFrame({'image_id': ids, 'cluster': kmeans.labels_})
df.to_csv('split.csv')
print(df.groupby('cluster').count())
clusters = kmeans.labels_
