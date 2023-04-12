import os

import numpy as np
import pandas as pd
import torch
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
# from pycocotools import coco
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


model = models.resnet50(pretrained=True)

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def extract_imagenet_features(img_path, transform=transformations, model=model):
    # Load the pre-trained ResNet50 model

    # Set the model to evaluation mode
    model.eval()

    # Define the image pre-processing transforms

    # Load the image and apply the pre-processing transforms
    img = Image.open(img_path)
    # img = np.array(img)/255.0
    # img = torch.from_numpy(img).float()
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    # Extract features from the image using the model
    with torch.no_grad():
        features = model(img_tensor)

    # Flatten the features tensor
    flattened_features = features.flatten()

    return flattened_features


class VisDroneDataset(Dataset):
    def __init__(self, data_dir='./datasets/visdrone/yolo_format/train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(data_dir, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        image = extract_imagenet_features(img_path=os.path.join(
            os.path.join(self.data_dir, 'images'
                         ),
            self.images[idx]), transform=self.transform)
        # if self.transform:
        #     image = self.transform(image)

        return image, self.images[idx]


# Define the transformations to be applied to the images
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = VisDroneDataset(transform=transformations)

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

kmeans = KMeans(n_clusters=100).fit(features, )
df = pd.DataFrame({'image_id': ids, 'cluster': kmeans.labels_})
df.to_csv('split.csv')
print(df.groupby('cluster').count())
clusters = kmeans.labels_