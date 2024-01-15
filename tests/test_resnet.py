import pytest


def test_train_eval_resnet():
    """Just to make sure the code can run"""
    hyperparameters = {"data_size_train": 128,
                       "data_size_test": 128,  "num_epochs": 1}
    from ResNet.train_and_eval_resnet import fine_tune_resnet, evaluate_resnet, device

    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.to(device)
    fine_tune_resnet(model, hyperparameters)
    average_inference_time, final_accuracy = evaluate_resnet(
        model, hyperparameters)
    # Just test that the code can run
    assert final_accuracy >= 0.0
