import torch
import torch_pruning as tp
from ResNet.train_and_eval_resnet import fine_tune_resnet, evaluate_resnet, train_and_evaluate_resnet
import os
import pandas as pd
from ResNet.variables import ROOT_PATH
from math import isclose
import numpy as np


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


def query_profiled_result(prune_ratio, profile_csv_file_name=None):
    """Query the profiled result from the profile_csv_file, return accuracy and latency"""
    if profile_csv_file_name:
        path = os.path.join(ROOT_PATH, "profile_data", profile_csv_file_name)
        df = pd.read_csv(path)
        df.columns = ['prune_ratio', 'accuracy', 'latency']
        row = df[np.isclose(df.iloc[:, 0], prune_ratio, rtol=0.01)]
        if not row.empty:
            return row.iloc[0]['accuracy'], row.iloc[0]['latency']
    return None, None


def prune_resnet_and_fine_tune(model, prune_ratio, hyperparameters, profile_csv_file_name=None):
    query_accuracy, query_latency = query_profiled_result(
        prune_ratio, profile_csv_file_name)
    if query_accuracy and query_latency:
        return query_accuracy, query_latency

    if prune_ratio == 0.0:
        return train_and_evaluate_resnet(model, hyperparameters)
    model = prune_resnet_with_tp(model, prune_ratio)
    return train_and_evaluate_resnet(model, hyperparameters)
