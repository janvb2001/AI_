import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os


class droneDataSet(Dataset):
    def __init__(self, dataset_path, prefix="", h5=False, augment=False):
        self.h5 = h5
        self.augment = augment
        if self.h5:
            self.file = h5py.File(dataset_path, "r")
            self.images = self.file["images"]
            self.targets = self.file["targets"]
        else:
            self.images = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "images.npy"))
            )
            self.targets = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "targets.npy"))
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.h5:
            images, targets = (
                torch.tensor(self.images[idx]),
                torch.tensor(self.targets[idx]),
            )
        else:
            images, targets = self.images[idx], self.targets[idx]

        if self.augment:
            # implement data augmentation
            None

        return images, targets
