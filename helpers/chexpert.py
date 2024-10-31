"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 18/10/24
"""
import torch
import torch.utils.data
import json
import os
from torchvision.datasets.folder import default_loader

# Download CheXpert-5x200 labels from https://raw.githubusercontent.com/QtacierP/PRIOR/refs/heads/main/datasets/zero_shot_classification/chexpert_5x200.json


class CheXpert_5x200(torch.utils.data.Dataset):
    def __init__(self, root, transform, loader=default_loader):
        super().__init__()

        self.root = root
        self.transform = transform
        self.loader = loader

        with open(os.path.join(root, "chexpert_5x200.json")) as f:
            self.labels = json.load(f)
            self.keys = list(self.labels.keys())

        print("Loaded", len(self.labels), "labels")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        key = self.keys[index]
        entry = self.labels[key]

        image_path = os.path.join(self.root, entry["image_path"])
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = entry["label"]
        # convert 1-hot to label index
        label = torch.tensor(label).argmax().item()

        return image, label



