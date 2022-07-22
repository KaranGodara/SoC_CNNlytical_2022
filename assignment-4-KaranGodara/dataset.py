import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torchvision.transforms.functional as TF

from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import torchvision

from tqdm import tqdm
import torch.optim as optim

import torchvision.transforms as transforms

# define your dataset class


class CarvanaDataset(Dataset):
    def __init__(self, X, y, transform=None):
        X = torch.from_numpy(X)
        X = torch.permute(X, (0, 3, 1, 2))

        y = torch.from_numpy(y)

        self.X = X
        self.y = y
        self.transform = transform

        # self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):

        image = self.X[index, :, :, :]
        mask = self.y[index, :, :]
        # assert type(image) != <class 'numpy.ndarray'>

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
