import os

from prune_resnet import prune_resnet_and_fine_tune
from RecordIO.RecordIO import save_to_file
from ResNet.variables import ROOT_PATH
from ResNet.Hyperparameters import Hyperparameters, get_hp_formal_cifar10, get_hp_test_cifar10, get_hp_test_cifar10_fast

from torchvision.models import resnet18


def create_csv_file(file_name):
    file_path = os.path.join(ROOT_PATH, "profile_data",
                             file_name+get_date()+".csv")
    with open(file_path, 'w') as file:
        file.write("Pruning_ratio, Accuracy, Latency\n")
    return file_path


def get_date():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def profile_with_pruning_ratio(model, hyperparameters, record_file_name, granularity=0.1):
    """Profile the model with pruning ratio, and save it to record_file_name"""

    file_path = create_csv_file(record_file_name)
    for pruning_ratio in range(0, 100, int(granularity*100)):
        pruning_ratio = pruning_ratio / 100.0
        accuracy, latency = prune_resnet_and_fine_tune(
            model, pruning_ratio, hyperparameters)
        print(
            f"Pruning ratio: {pruning_ratio}, accuracy: {accuracy}, latency: {latency}")
        save_to_file([pruning_ratio], [accuracy, latency], file_path)


if __name__ == "__main__":
    # hyperparameters = get_hp_test_cifar10_fast()
    hyperparameters = get_hp_formal_cifar10()
    hyperparameters.training_noise = 0.1
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    # model = resnet18(pretrained=True)
    profile_with_pruning_ratio(model, hyperparameters,
                               "profile_test_fast_resnet18_pruning_ratio_0.1_noise")
