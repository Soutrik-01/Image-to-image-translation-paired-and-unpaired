from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class X_Y_Dataset(Dataset):
    def __init__(self, root_X, root_Y, transform=None):
        self.root_x = root_X
        self.root_y = root_Y
        self.transform = transform

        self.x_images = os.listdir(root_X)
        self.y_images = os.listdir(root_Y)
        self.length_dataset = max(len(self.x_images), len(self.y_images)) # 1000, 1500
        self.x_len = len(self.x_images)
        self.y_len = len(self.y_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        x_img = self.x_images[index % self.x_len]
        y_img = self.y_images[index % self.y_len]

        x_path = os.path.join(self.root_x, x_img)
        y_path = os.path.join(self.root_y, y_img)

        x_img = np.array(Image.open(x_path).convert("RGB"))
        y_img = np.array(Image.open(y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=x_img, image0=y_img)
            x_img = augmentations["image"]
            y_img = augmentations["image0"]

        return x_img, y_img




