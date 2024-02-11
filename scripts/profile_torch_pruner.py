

from TorchPruner.prune_resnet import prune_resnet_and_fine_tune
from RecordIO.RecordIO import save_to_file, get_csv_file, query_profiled_result, find_record
from ResNetTrain.variables import ROOT_PATH
from ResNetTrain.Hyperparameters import Hyperparameters, get_hp_formal_cifar10, get_hp_test_cifar10, get_hp_test_cifar10_fast, get_hp_formal_cifar10_resnet50

from torchvision.models import resnet18, resnet34, resnet50
from RecordIO.WritingInfo import WritingInfo, get_output_file_name
import argparse


def profile_with_pruning_ratio(model, hyperparameters_model, model_name,
                               granularity=0.1):
    """
    Profile the model with different pruning ratioes, and save it to profile_data folder
    """
    training_poison_chance = hyperparameters_model.training_poison_chance
    testing_data_noise = hyperparameters_model.testing_data_noise
    record_file_name = get_output_file_name(
        model_name=model_name, testing_data_noise=testing_data_noise, training_poison_chance=training_poison_chance)
    file_path = get_csv_file(record_file_name)
    for pruning_ratio in range(0, 100, int(granularity*100)):
        pruning_ratio = pruning_ratio / 100.0
        accuracy, latency = find_record(
            model_name, testing_data_noise, training_poison_chance, pruning_ratio)
        if not accuracy and not latency:
            writing_info = WritingInfo(
                model_name=model_name, testing_data_noise=testing_data_noise, training_poison_chance=training_poison_chance)
            hyperparameters_model.pruning_ratio = pruning_ratio
            accuracy, latency = prune_resnet_and_fine_tune(
                model, pruning_ratio, hyperparameters_model, writing_info)
            # save_to_file([pruning_ratio], [accuracy, latency], file_path)
        print(
            f"Pruning ratio: {pruning_ratio}, accuracy: {accuracy}, latency: {latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile Torch Pruner')
    parser.add_argument('--training_poison_chance', type=float, default=0.0,
                        help='Training poison level (randomly change output label)')
    parser.add_argument('--testing_data_noise', type=float, default=0.1,
                        help='Testing noise level (adding Gaussian noise to input image,\
                              training input is not influenced)')
    args = parser.parse_args()
    # hyperparameters = get_hp_test_cifar10_fast()
    hyperparameters = get_hp_formal_cifar10()
    # hyperparameters = get_hp_formal_cifar10_resnet50()

    hyperparameters.training_poison_chance = args.training_poison_chance
    hyperparameters.testing_data_noise = args.testing_data_noise
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    model.model_name = "resnet18"
    # model = resnet50(weights='ResNet50_Weights.DEFAULT')
    profile_with_pruning_ratio(
        model, hyperparameters, model.model_name)
