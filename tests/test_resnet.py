import pytest
from ResNet.Hyperparameters import Hyperparameters


def test_train_eval_resnet():
    """Just to make sure the code can run"""
    hyperparameters = Hyperparameters()
    hyperparameters.data_size_train = 128
    hyperparameters.data_size_test = 128
    hyperparameters.num_epochs = 1
    from ResNet.train_and_eval_resnet import fine_tune_resnet, evaluate_resnet, device

    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.to(device)
    fine_tune_resnet(model, hyperparameters)
    final_accuracy, average_inference_time = evaluate_resnet(
        model, hyperparameters)
    # Just test that the code can run
    assert final_accuracy >= 0.0
