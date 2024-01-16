
from torchvision.models import resnet18
import pytest

from ResNet.Hyperparameters import get_hp_formal_cifar10
from Optimization.automl_binary_search import automl_bs_find_pruning_ratio


def test_automl_bs_find_pruning_ratio1():
    resnet18_profile_file = "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv"
    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(pretrained=True)

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 50, profile_csv_file_name=resnet18_profile_file)
    assert pruning_ratio == pytest.approx(
        0.7), "test_automl_bs_find_pruning_ratio failed"

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 20, profile_csv_file_name=resnet18_profile_file)
    assert pruning_ratio == pytest.approx(
        0.9), "test_automl_bs_find_pruning_ratio failed"


def test_automl_bs_find_pruning_ratio2():
    resnet18_profile_file = "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv"
    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(pretrained=True)

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 95, profile_csv_file_name=resnet18_profile_file)
    assert pruning_ratio == pytest.approx(
        0.1), "test_automl_bs_find_pruning_ratio failed"

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 99, profile_csv_file_name=resnet18_profile_file)
    assert pruning_ratio == None, "test_automl_bs_find_pruning_ratio failed"
