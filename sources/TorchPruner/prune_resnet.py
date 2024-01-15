import torch
import torch_pruning as tp


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
