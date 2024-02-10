import pytest
from ResNetTrain.Hyperparameters import Hyperparameters, get_hp_test_cifar10_fast


def test_train_eval_resnet():
    """Just to make sure the code can run"""
    hyperparameters = get_hp_test_cifar10_fast()
    from ResNetTrain.train_and_eval_resnet import fine_tune_resnet, evaluate_resnet, device

    from torchvision.models import resnet18
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    model.to(device)
    fine_tune_resnet(model, hyperparameters)
    final_accuracy, average_inference_time = evaluate_resnet(
        model, hyperparameters)
    # Just test that the code can run
    assert final_accuracy >= 0.0
