
from torchvision.models import resnet18
import pytest
import os

from ResNetTrain.Hyperparameters import get_hp_formal_cifar10
from Optimization.automl_binary_search import automl_bs_find_pruning_ratio
from RecordIO.WritingInfo import WritingInfo
import shutil
from ResNetTrain.variables import ROOT_PATH


def generate_data_csv_file():
    source_file = os.path.join(
        ROOT_PATH, 'profile_data/profile_data_for_test_automl.csv')
    destination_file = os.path.join(
        ROOT_PATH, 'profile_data/profile_data_resnet18_noise_0.0_poison_0.0.csv')
    if os.path.exists(destination_file):
        copied_intermediate_file = os.path.join(
            ROOT_PATH, 'profile_data/copied_intermediate_file.csv')
        shutil.copyfile(destination_file, copied_intermediate_file)

    shutil.copyfile(source_file, destination_file)


def recover_copied_data_file():
    copied_intermediate_file = os.path.join(
        ROOT_PATH, 'profile_data/copied_intermediate_file.csv')
    destination_file = os.path.join(
        ROOT_PATH, 'profile_data/profile_data_resnet18_noise_0.0_poison_0.0.csv')
    if os.path.exists(copied_intermediate_file):
        shutil.copyfile(copied_intermediate_file, destination_file)


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio1():
    generate_data_csv_file()
    writing_info = WritingInfo(
        testing_data_noise=0.0, model_name="resnet18", training_poison_chance=0.0)
    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(weights='ResNet18_Weights.DEFAULT')

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 50, writing_info=writing_info)
    assert pruning_ratio == pytest.approx(
        0.7), "test_automl_bs_find_pruning_ratio failed"

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 20, writing_info=writing_info)
    assert pruning_ratio == pytest.approx(
        0.9), "test_automl_bs_find_pruning_ratio failed"
    recover_copied_data_file()


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio2():
    generate_data_csv_file()

    writing_info = WritingInfo(
        testing_data_noise=0.0, model_name="resnet18", training_poison_chance=0.0)

    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(weights='ResNet18_Weights.DEFAULT')

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 95, writing_info=writing_info)
    assert pruning_ratio == pytest.approx(
        0.1), "test_automl_bs_find_pruning_ratio failed"

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 99, writing_info=writing_info)
    assert pruning_ratio == None, "test_automl_bs_find_pruning_ratio failed"
    recover_copied_data_file()


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio3():
    generate_data_csv_file()

    writing_info = WritingInfo(
        testing_data_noise=0.0, model_name="resnet18", training_poison_chance=0.0)

    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(weights='ResNet18_Weights.DEFAULT')

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 85, writing_info=writing_info)
    assert pruning_ratio == pytest.approx(
        0.4), "test_automl_bs_find_pruning_ratio failed"
    recover_copied_data_file()
