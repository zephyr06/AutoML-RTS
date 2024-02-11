import pytest
import torch
from ResNetTrain.Cifar10DataIO import data_loader, exam_and_prepare_noised_dataset, get_path_with_noise

batch_size = 64


def test_data_loader():
    data_size = 100
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


def test_data_loader_add_poison():
    train_data_size = 100
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=train_data_size,
                                             test_data_size=10, training_poison_chance=0.5)
    label_sum_w_noise = 0
    for i, (images, labels) in enumerate(train_loader):
        label_sum_w_noise += labels.sum().item()

    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             test_only=False,
                                             training_data_size=train_data_size,
                                             test_data_size=10, training_poison_chance=0)
    label_sum_wo_noise = 0
    for i, (images, labels) in enumerate(train_loader):
        label_sum_wo_noise += labels.sum().item()
    assert label_sum_wo_noise != label_sum_w_noise


def test_data_loader_with_noise():
    data_dir = './data'
    noise0 = 0.0
    noise2 = 0.2
    exam_and_prepare_noised_dataset(data_dir, noise0)
    testing_path_noise0 = get_path_with_noise(
        data_dir, noise0, test_only=True)
    test_dataset0 = torch.load(testing_path_noise0)
    mean0 = torch.mean(test_dataset0[0][0])

    exam_and_prepare_noised_dataset(data_dir, noise2)
    testing_path_noise2 = get_path_with_noise(
        data_dir, noise2, test_only=True)
    test_dataset2 = torch.load(testing_path_noise2)
    mean2 = torch.mean(test_dataset2[0][0])
    assert pytest.approx(mean0+noise2-noise0) == mean2
