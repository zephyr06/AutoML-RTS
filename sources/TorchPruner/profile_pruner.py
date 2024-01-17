

from prune_resnet import prune_resnet_and_fine_tune
from RecordIO.RecordIO import save_to_file, get_csv_file, query_profiled_result, find_record
from ResNet.variables import ROOT_PATH
from ResNet.Hyperparameters import Hyperparameters, get_hp_formal_cifar10, get_hp_test_cifar10, get_hp_test_cifar10_fast

from torchvision.models import resnet18
from RecordIO.WritingInfo import  WritingInfo, get_output_file_name




def profile_with_pruning_ratio(model, hyperparameters_model, model_name, training_data_noise,\
                                granularity=0.1):
    """
    Profile the model with different pruning ratioes, and save it to profile_data folder
    """
    record_file_name = get_output_file_name(model_name=model_name, training_data_noise=training_data_noise)
    file_path = get_csv_file(record_file_name)
    for pruning_ratio in range(0, 100, int(granularity*100)):
        pruning_ratio = pruning_ratio / 100.0
        accuracy, latency = find_record(model_name, training_data_noise, pruning_ratio)
        if not accuracy and not latency:
            accuracy, latency = prune_resnet_and_fine_tune(
                model, pruning_ratio, hyperparameters_model)
            save_to_file([pruning_ratio], [accuracy, latency], file_path)
        print(
            f"Pruning ratio: {pruning_ratio}, accuracy: {accuracy}, latency: {latency}")
    

if __name__ == "__main__":
    # hyperparameters = get_hp_test_cifar10_fast()
    hyperparameters = get_hp_formal_cifar10()
    hyperparameters.training_noise = 0.1
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    # model = resnet18(pretrained=True)
    profile_with_pruning_ratio(model, hyperparameters,
                               "resnet18", "0.0")
