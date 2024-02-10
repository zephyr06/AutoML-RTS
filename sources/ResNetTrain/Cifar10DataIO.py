import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
import random
import os

# from .variables import ROOT_PATH


class AddGaussianNoise:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def load_dataset(data_dir, data_size, test_only=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=~(test_only),
        download=True, transform=transform,
    )
    train_dataset = Subset(train_dataset, list(range(data_size)))
    return train_dataset


def get_mean_std(dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), num_workers=2)
    data = next(iter(loader))
    images, _ = data
    mean = torch.mean(images, dim=(0, 2, 3))
    std = torch.std(images, dim=(0, 2, 3))
    return mean, std


def add_noise_to_data_loader(data_loader, noise_level):
    pass


def get_path_with_noise(data_dir, noise, test_only=False):
    return os.path.join(data_dir, f"noise_{noise}"+("_test" if test_only else "")+".pth")


def exam_and_prepare_noised_dataset(data_dir, noise):
    training_path = get_path_with_noise(data_dir, noise, test_only=False)
    testing_path = get_path_with_noise(data_dir, noise, test_only=True)
    if os.path.exists(training_path) and os.path.exists(testing_path):
        return
    transform_w_noise = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=noise, std=noise),
        # normalize,
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=None,
    )
    torch.save(train_dataset, training_path)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform_w_noise,
    )
    torch.save(test_dataset, testing_path)


def data_loader_noise(data_dir,
                      batch_size,
                      random_seed=42,
                      valid_size=0.1,
                      shuffle=True,
                      test_only=False,
                      training_data_size=1000,
                      test_data_size=1000,
                      training_poison_chance=0.0,
                      training_data_noise=0.0,
                      testing_data_noise=0.0):
    exam_and_prepare_noised_dataset(data_dir, testing_data_noise)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # AddGaussianNoise(mean=training_data_noise, std=training_data_noise),
        # normalize,
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=transform,
    )
    a = 1
    return train_dataset


def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test_only=False,
                training_data_size=1000,
                test_data_size=1000,
                training_poison_chance=0.0,
                training_data_noise=0.0,
                testing_data_noise=0.0):
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
        AddGaussianNoise(mean=training_data_noise, std=training_data_noise),
        # normalize,
    ])

    # TODO: consider save the dataset with poison
    def add_target_poison(label):
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
        download=True, transform=transform, target_transform=add_target_poison
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


if __name__ == "__main__":
    data_size = 100
    train_loader = data_loader_noise(data_dir='./data',
                                     batch_size=data_size,
                                     test_only=False,
                                     training_data_size=data_size,
                                     test_data_size=data_size)
