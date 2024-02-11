import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
import random
import os

import matplotlib.pyplot as plt
from ResNetTrain.variables import ROOT_PATH


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


def normalize_dataset(dataset):
    mean, std = get_mean_std(dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset.transform = transform
    return dataset


def get_path_with_noise(data_dir, noise, test_only=False):
    return os.path.join(data_dir, f"noise_{noise}"+("_test" if test_only else "_train")+".pth")


def exam_and_prepare_noised_dataset(data_dir, noise):
    training_path = get_path_with_noise(data_dir, noise, test_only=False)
    testing_path = get_path_with_noise(data_dir, noise, test_only=True)
    if os.path.exists(training_path) and os.path.exists(testing_path):
        return
    transform_wo_noise = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transform_w_noise = transforms.Compose([
        transform_wo_noise,
        AddGaussianNoise(mean=noise, std=noise),
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform_wo_noise,
    )
    torch.save(train_dataset, training_path)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform_w_noise,
    )
    torch.save(test_dataset, testing_path)


def show_cifar10_image(dataset, index=1):
    plt.imshow(dataset[index][0].permute(1, 2, 0))
    plt.show()


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
    training_path = get_path_with_noise(
        data_dir, training_data_noise, test_only=False)
    testing_path = get_path_with_noise(
        data_dir, testing_data_noise, test_only=True)

    # Load the transformed training dataset
    train_dataset = torch.load(training_path)
    normalize_dataset(train_dataset)

    # Load the transformed testing dataset
    test_dataset = torch.load(testing_path)
    normalize_dataset(train_dataset)

    # train_dataset = datasets.CIFAR10(
    #     root=os.path.dirname(training_path), train=True,
    #     download=False, transform=transform,
    # )

    # )
    a = 1
    return train_dataset, test_dataset


def get_test_dataloader(data_dir,
                        batch_size,
                        transform,
                        random_seed=42,
                        shuffle=True,
                        test_data_size=1000,
                        testing_data_noise=0.0):
    exam_and_prepare_noised_dataset(data_dir, testing_data_noise)
    testing_path = get_path_with_noise(
        data_dir, testing_data_noise, test_only=True)
    test_dataset = datasets.CIFAR10(
        root=os.path.dirname(testing_path), train=False,
        download=False, transform=transform)
    test_dataset = Subset(test_dataset, list(range(test_data_size)))
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )


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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if test_only:
        return get_test_dataloader(data_dir, batch_size, transform,
                                   random_seed, shuffle, test_data_size, testing_data_noise)
    else:  # do not add noise to training data currently
        def add_target_poison(label):
            if random.random() < training_poison_chance:
                # Return a random label between 0 and 9
                return random.randint(0, 9)
            else:
                return label
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
    train_loader, test_loader = data_loader_noise(data_dir='./data',
                                                  batch_size=data_size,
                                                  test_only=False,
                                                  training_data_size=data_size,
                                                  test_data_size=data_size,
                                                  testing_data_noise=0.2)
    # train_loader, test_loader = data_loader(data_dir='./data',
    #                                         batch_size=data_size,
    #                                         test_only=False,
    #                                         training_data_size=data_size,
    #                                         test_data_size=data_size,
    #                                         testing_data_noise=0)

    print(len(train_loader))
    print(len(test_loader))
