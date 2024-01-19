import torch
import torch_pruning as tp
from ResNet.train_and_eval_resnet import fine_tune_resnet, evaluate_resnet, train_and_evaluate_resnet
from ResNet.variables import ROOT_PATH

from RecordIO.RecordIO import query_profiled_result, find_record, get_csv_file, save_to_file
from RecordIO.WritingInfo import WritingInfo, get_output_file_name


def prune_resnet_with_tp(model, prune_ratio=0.5):
    model.to('cpu')

    example_inputs = torch.randn(1, 3, 224, 224)
    # 1. Importance criterion
    # or GroupNormImportance(p=2), GroupHessianImportance(), etc.
    imp = tp.importance.GroupTaylorImportance()

    # 2. Initialize a pruner with the model and the importance criterion
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        # e.g., remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        pruning_ratio=prune_ratio,
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )

    # 3. Prune & finetune the model
    base_macs, base_nparams = tp.utils.count_ops_and_params(
        model, example_inputs)
    if isinstance(imp, tp.importance.GroupTaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        # A dummy loss, please replace this line with your loss function and data!
        loss = model(example_inputs).sum()
        loss.backward()  # before pruner.step()

    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    return model


# TODO: test whether this function correctly saves the result to file
def prune_resnet_and_fine_tune(model, prune_ratio, hyperparameters, writing_info=None):

    if writing_info:
        query_accuracy, query_latency = find_record(
            writing_info.model_name, writing_info.training_data_noise, prune_ratio)
        if query_accuracy and query_latency:
            return query_accuracy, query_latency
    print("Performing pruning and fine-tuning...")
    if prune_ratio == 0.0:
        accuracy_fine_tuned, latency_fine_tuned = train_and_evaluate_resnet(
            model, hyperparameters)
    else:
        model = prune_resnet_with_tp(model, prune_ratio)
        accuracy_fine_tuned, latency_fine_tuned = train_and_evaluate_resnet(
            model, hyperparameters)

    # save the result to file
    profile_file_name = get_output_file_name(
        writing_info.model_name, writing_info.training_data_noise)
    profile_csv_file_path = get_csv_file(profile_file_name)
    save_to_file([prune_ratio], [accuracy_fine_tuned,
                 latency_fine_tuned], profile_csv_file_path)

    return accuracy_fine_tuned, latency_fine_tuned
