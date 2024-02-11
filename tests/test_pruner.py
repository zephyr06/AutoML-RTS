import pytest

import torch
from torchvision.models import resnet18
import torch_pruning as tp
from TorchPruner.prune_resnet import prune_resnet_with_tp, query_profiled_result


@pytest.mark.timeout(1)
def test_pruner():
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    initial_layer = model.layer1[0].conv1.out_channels
    model = prune_resnet_with_tp(model, 0.5)
    pruned_layer = model.layer1[0].conv1.out_channels
    assert initial_layer == 2*pruned_layer


@pytest.mark.timeout(1)
def test_query_profiled_result():
    accuracy, latency = query_profiled_result(
        0.0, "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv")
    assert accuracy == pytest.approx(95.3)
    assert latency == pytest.approx(1.324)

    accuracy, latency = query_profiled_result(
        0.5, "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv")
    assert accuracy == pytest.approx(84.74)
    assert latency == pytest.approx(0.963)

    accuracy, latency = query_profiled_result(
        0.55, "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv")
    assert accuracy == None
    assert latency == None
