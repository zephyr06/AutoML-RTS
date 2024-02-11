import seaborn as sns
import numpy as np
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from Optimization.automl_binary_search import automl_bs_find_pruning_ratio
from utils import read_df_from_csv, marker_styles, get_all_profile_files
from ResNetTrain.Hyperparameters import get_hp_formal_cifar10
from RecordIO.WritingInfo import WritingInfo


def generate_noise_vector(file_num):
    noise_seq = np.linspace(0, 0.1 * file_num, file_num, endpoint=False)
    for i in range(file_num):
        noise_seq[i] = round(noise_seq[i], 1)
    return noise_seq


def draw_noise_latency_curve(perf_requirement):

    csv_files = get_all_profile_files("profile_data_resnet18_noise_0.*.csv")
    file_num = len(csv_files)
    colors = sns.color_palette('husl', n_colors=1)

    # prepare parameters to call automl
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    hyperparameters = get_hp_formal_cifar10()
    writing_info = WritingInfo()
    writing_info.model_name = "resnet18"

    noise_seq = generate_noise_vector(file_num)
    latency_seq = np.zeros(file_num)
    for i in range(file_num):
        df = read_df_from_csv(csv_files[i])
        noise_curr = noise_seq[i]
        writing_info.testing_data_noise = noise_curr
        target_pruning_ratio, target_accuracy, target_latency = automl_bs_find_pruning_ratio(
            model, hyperparameters, None, None, perf_requirement, writing_info=writing_info)
        latency_seq[i] = target_latency
        print("noise: ", noise_curr, "pruning ratio: ",
              target_pruning_ratio, "latency: ", latency_seq[i])

    colors = sns.color_palette('husl', n_colors=1)
    fig, ax1 = plt.subplots()
    sns.lineplot(x=noise_seq, y=latency_seq, marker=marker_styles[0])
    plt.xticks(noise_seq)
    ax1.set_xlabel("Noise level")
    ax1.set_ylabel("Latency (s)")
    ax1.grid(True, linestyle='--')  # Add dashed grid lines
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    perf_requirement = 92
    draw_noise_latency_curve(perf_requirement)
