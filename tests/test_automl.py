
from torchvision.models import resnet18
import pytest

from ResNetTrain.Hyperparameters import get_hp_formal_cifar10
from Optimization.automl_binary_search import automl_bs_find_pruning_ratio
from RecordIO.WritingInfo import WritingInfo


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio1():
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


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio2():

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


@pytest.mark.timeout(1)
def test_automl_bs_find_pruning_ratio3():

    writing_info = WritingInfo(
        testing_data_noise=0.0, model_name="resnet18", training_poison_chance=0.0)

    hyperparameters = get_hp_formal_cifar10()
    model = resnet18(weights='ResNet18_Weights.DEFAULT')

    pruning_ratio, accuracy, latency = automl_bs_find_pruning_ratio(
        model, hyperparameters, None, None, 85, writing_info=writing_info)
    assert pruning_ratio == pytest.approx(
        0.4), "test_automl_bs_find_pruning_ratio failed"
