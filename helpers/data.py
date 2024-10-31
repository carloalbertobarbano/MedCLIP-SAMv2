"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 08/05/24
"""
import json
import clip
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import datasets
from torchvision.datasets import utils as datasets_utils
from glob import glob


class CaptionDataset(Dataset):
    def __init__(self, root, transform, caption_file, tokenize=clip.tokenize,
                 definition_caption=False, samples_per_class=None, num_classes=None, device="cuda"):
        """
        Args:
            root (string): Root directory of dataset.
            transform (callable): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            caption_file (string): Json file containing the captions.
            tokenize (callable, optional): A function/transform that takes in a caption and returns a list of tokens.
            definition_caption (bool): If True, preprend each caption with "A <xxx> is a <caption>"
                                       where <xxx> is the class name.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.caption_file = caption_file
        self.tokenize = tokenize
        self.definition_caption = definition_caption
        self.num_classes = num_classes
        self.classes = self.load_classes(caption_file, num_classes)
        self.samples_per_class = samples_per_class
        self.images, self.captions = self.load_image_list(root, self.classes, tokenize,
                                                          definition_caption, samples_per_class)
        self.device = device

    def load_classes(self, caption_file, num_classes=None):
        with open(caption_file, "r") as f:
            captions = json.load(f)

        classes = []
        for class_name, caption_list in captions.items():
            classes.append(class_name)

        if num_classes is not None:
            # sample num_classes randomly
            print("Sampling", num_classes, "classes")
            idx = torch.randperm(len(classes))
            classes = [classes[i] for i in idx]
            classes = classes[:num_classes]
            print("Selected classes:", classes)

        return classes

    def load_image_list(self, root, classes, tokenize=clip.tokenize, definition_caption=False,
                        samples_per_class=None):
        image_list = []
        captions = defaultdict(list)

        for class_name in classes:
            generation_file = os.path.join(root, class_name, "images.txt")
            if not os.path.isfile(generation_file):
                print("Skipping class", class_name, "as images.txt is missing")
                continue

            with open(generation_file, "r") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if line]  # discard empty

            image_names = [os.path.basename(line.split("|")[0]) for line in lines]
            prompts = [line.split("|")[1] for line in lines]
            if definition_caption:
                prompts = [f"A {class_name} is {p[0].lower()}{p[1:]}" for p in prompts]
            if tokenize is not None:
                prompts = tokenize(prompts)

            if samples_per_class is not None:
                image_names = image_names[:samples_per_class]
                prompts = prompts[:samples_per_class]

            image_list.extend([(class_name, os.path.join(root, class_name, image_name)) for image_name in image_names])
            captions[class_name] = prompts

        return image_list, captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        class_name, image_path = self.images[index]

        if ".png" in image_path:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.load(image_path, map_location=self.device)
            image.requires_grad = False

        # Pick a random caption of the same class
        caption_idx = torch.randint(len(self.captions[class_name]), (1,)).item()
        caption = self.captions[class_name][caption_idx]

        return image, caption


class CocoImages(datasets.CocoCaptions):
    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        return image


if __name__ == '__main__':
    import sys
    import clip

    if len(sys.argv) < 3:
        print("Usage: python -m helpers.data <root> <captions.json>")
        sys.exit(1)

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    dataset = CaptionDataset(root=sys.argv[1], transform=preprocess, caption_file=sys.argv[2], device="cpu")
    print(f"Loaded {len(dataset)} images")

    image, caption = dataset[torch.randint(len(dataset), (1,)).item()]
    print(f"Image shape: {image.shape}")
    print(f"Caption shape: {caption.shape}")





