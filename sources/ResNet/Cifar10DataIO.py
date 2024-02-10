import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
import random


def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test_only=False,
                training_data_size=1000,
                test_data_size=1000,
                training_poison_chance=0.0):
    """Load the CIFAR10 dataset and perform preprocessing, with simple data size assignments for the convenience of development."""

    random.seed(random_seed)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # TODO: consider save the dataset with noise
    def add_target_noise(label):
        if random.random() < training_poison_chance:
            # Return a random label between 0 and 9
            return random.randint(0, 9)
        else:
            return label

    if test_only:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
        dataset = Subset(dataset, list(range(test_data_size)))

        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform, target_transform=add_target_noise
    )
    train_dataset = Subset(train_dataset, list(range(training_data_size)))

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )
    valid_dataset = Subset(valid_dataset, list(range(training_data_size)))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)
