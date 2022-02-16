"""Helper module for loading train and test dataloaders."""

import numpy.random
import torch, torchvision
from torch.utils.data import Subset
import torchvision.transforms as transforms

DATA_FOLDER = "../data/"

def get_dataloader(batch_size, train_size=None, test_size=None,
                   transform_train_data=True, add_noise=0,
                   drop_pixels=0, shuffle_pixels=0,
                   data_folder=DATA_FOLDER):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
        corrupt_data
        add_noise: Std dev of Gaussian noise to add to the data (per sample basis)
        drop_pixels: Percentage of pixels to randomly drop.
        data_folder: Where the data is stored when downloaded.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    all_transforms = [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ] if transform_train_data else [transforms.ToTensor(), normalize]

    if add_noise: all_transforms.append(RandomNoise(add_noise))
    if drop_pixels: all_transforms.append(RandomDrop(drop_pixels))
    if shuffle_pixels: all_transforms.append(ShufflePixels(shuffle_pixels))

    transform = transforms.Compose(all_transforms)

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, transform=test_transform, download=True
    )

    if train_size:
        indices = numpy.random.permutation(numpy.arange(len(train_dataset)))
        train_dataset = Subset(train_dataset, indices[:train_size])

    if test_size:
        indices = numpy.random.permutation(numpy.arange(len(test_dataset)))
        test_dataset = Subset(train_dataset, indices[:test_size])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader