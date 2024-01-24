import pytest

from ResNet.Cifar10DataIO import data_loader

batch_size = 64


def test_data_loader():
    data_size=100
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=data_size,
                                             test_data_size=data_size)
    assert data_size == len(train_loader.dataset)
    assert data_size == len(valid_loader.dataset)
    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              test_only=True, test_data_size=data_size)
    assert data_size == len(test_loader.dataset)


def test_data_loader_add_noise():
    train_data_size = 100
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=train_data_size,
                                             test_data_size=10, training_noise_chance=0.5)
    label_sum_w_noise = 0
    for i, (images, labels) in enumerate(train_loader):
        label_sum_w_noise += labels.sum().item()

    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=train_data_size,
                                             test_data_size=10, training_noise_chance=0)
    label_sum_wo_noise = 0
    for i, (images, labels) in enumerate(train_loader):
        label_sum_wo_noise += labels.sum().item()
    assert label_sum_wo_noise != label_sum_w_noise
