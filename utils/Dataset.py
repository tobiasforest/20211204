# A reproduction using PyTorch on the paper:
# UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS
# https://hal-enpc.archives-ouvertes.fr/hal-01864755

import torch
import torchvision
import torchvision.transforms as transforms
# import torchvision.transforms.functional.rotate as rotate
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from config import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # CIFAR-10 dataset & dataloader
# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform)
# train_dataloader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]),
}

data_post_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),  # random error i guess
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
}


class RotationDataset(torchvision.datasets.CIFAR10):
    def __init__(self, split, root, preTransform=None, postTransform=None,
                 download=False):
        super(RotationDataset, self).__init__(
            root, train=(split == 'train'), transform=preTransform, download=download)
        self.split = split
        self.preTransform = preTransform
        self.postTransform = postTransform

    def __getitem__(self, index):
        img, _ = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90
        rot_img = transforms.functional.rotate(img, rot_angle)

        if self.postTransform:
            rot_img = self.postTransform(rot_img)
        else:
            rot_img = transforms.ToTensor(rot_img)

        # rot_img = np.expand_dims(rot_img, 0)
        rot_img = rot_img.unsqueeze(0)
        # rot_img = torch.unsqueeze(rot_img, 0)

        return rot_img, rot_class

    def __len__(self):
        return super(RotationDataset, self).__len__()


# train_dataset = RotationDataset(
#     root='./data', train=True, download=True, transform=transform)
# test_dataset = RotationDataset(
#     root='./data', train=False, download=True, transform=transform)
# train_dataloader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# dataloaders = {'train': train_dataloader, 'val': val_dataloader}

image_datasets = {x: RotationDataset(x, './data', data_pre_transforms[x], data_post_transforms[x])
                  for x in ['train', 'val']}

assert image_datasets

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True)
               for x in ['train', 'val']}


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


def imshow(img, title=None):
    """
    Show an example image.

    Args:
        img (Tensor): An image tensor.
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    inputs, labels = next(iter(dataloaders['train']))
    print(labels)
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x.item() for x in labels])
