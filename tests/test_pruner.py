import pytest

import torch
from torchvision.models import resnet18
import torch_pruning as tp
from TorchPruner.prune_resnet import prune_resnet_with_tp


def test_pruner():
    model = resnet18(pretrained=True)
    initial_layer = model.layer1[0].conv1.out_channels
    model = prune_resnet_with_tp(model, 0.5)
    pruned_layer = model.layer1[0].conv1.out_channels
    assert initial_layer == 2*pruned_layer
