import warnings

from TorchPruner.prune_resnet import prune_resnet_and_fine_tune
from ResNet.train_and_eval_resnet import train_and_evaluate_resnet
from RecordIO.WritingInfo import WritingInfo, get_output_file_name

def automl_bs_find_pruning_ratio(model, hyperparameters_model, train_data_loader,
                                 test_data_loader, perf_required: float, granularity=0.1, \
                                    writing_info : WritingInfo=None):
    """Perform binary search to find the maximum pruning ratio while satisfying the performance requirement.
    Pruning ratio refers to the portion of channels to be pruned.
    This function combines both profiling and optimization.
    perf_required: in percentage, e.g., 50 means 50%.

    Assumptions:
    - graularity is 0.1 by default;, smaller granularity means more accurate but slower;
    - maximum pruning ratio is 1-granularity, minimum pruning ratio is 0.00;
    - Performance and Latency are monotonically decreasing when the pruning ratio is increasing;
    """
    accuracy_no_prune, latency_no_prune = prune_resnet_and_fine_tune(
        model, 0.0, hyperparameters_model, writing_info)
    if accuracy_no_prune < perf_required:
        warnings.warn(
            f"Accuracy without pruning is {accuracy_no_prune}, which is lower than the required {perf_required}.")
        return None, None, None
    accuracy_max_prune, latency_max_prune = prune_resnet_and_fine_tune(
        model, 1-granularity, hyperparameters_model, writing_info)
    if accuracy_max_prune >= perf_required:
        return 1-granularity, accuracy_max_prune, latency_max_prune

    # Binary search
    prune_ratio_lower = 0.0
    prune_ratio_upper = 1-granularity
    final_accuracy = accuracy_no_prune
    final_latency = latency_no_prune
    while (prune_ratio_lower+granularity < prune_ratio_upper):
        prune_ratio = (prune_ratio_lower + prune_ratio_upper) / 2
        prune_ratio = int(prune_ratio/granularity) * granularity
        accuracy, latency = prune_resnet_and_fine_tune(
            model, prune_ratio, hyperparameters_model, writing_info)
        if accuracy >= perf_required:
            final_accuracy = accuracy
            final_latency = latency
            prune_ratio_lower = prune_ratio
        else:
            prune_ratio_upper = prune_ratio-granularity
    # exam whether prune_ratio_upper is the final result
    accuracy_upper, latency_upper = prune_resnet_and_fine_tune(
        model, prune_ratio_upper, hyperparameters_model, writing_info)
    if accuracy_upper >= perf_required:
        final_accuracy = accuracy_upper
        final_latency = latency_upper
        prune_ratio_lower = prune_ratio_upper
    return prune_ratio_lower, final_accuracy, final_latency
