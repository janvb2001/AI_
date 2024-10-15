import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import h5py
import hdf5plugin
import os
import torch.nn.functional as F


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
                torchvision.transforms.ToTensor()(self.images[idx]),
                # torch.tensor(np.array(self.images[idx]).Totensor()),
                torch.tensor(np.array(self.targets[str(idx).zfill(5)])),
            )

            images = torch.moveaxis(images, (1, 2), (0, 1))

            targets = targets.flatten()
            padding_length = 108 - targets.size(0)

            if padding_length > 0:
                targets = F.pad(targets, (0, padding_length), "constant", 0)
            else:
                targets = targets[:109]

        else:
            images, targets = self.images[idx], self.targets[idx]

        if self.augment:
            # implement data augmentation
            None

        return images, targets

