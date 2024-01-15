import pytest

from ResNet.Cifar10DataIO import data_loader

batch_size = 64


def test_data_loader():
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=1000,
                                             test_data_size=1000)
    assert 1000 == len(train_loader.dataset)
    assert 1000 == len(valid_loader.dataset)
    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              test_only=True, test_data_size=100)
    assert 100 == len(test_loader.dataset)
