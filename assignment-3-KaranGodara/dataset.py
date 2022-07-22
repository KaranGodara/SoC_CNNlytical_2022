import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.autograd import Variable

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing
%matplotlib inline

transform_train = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CIFAR_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=transform_train):
        'Initialization'
        # X for images, y for labels
        y = torch.from_numpy(y)
        y = y.squeeze(1)
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        label = self.y[index]
        image = torch.div(image, 255.0)
        return image, label
