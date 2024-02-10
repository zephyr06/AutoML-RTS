import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from ResNetTrain.variables import ROOT_PATH
import seaborn as sns
from utils import read_df_from_csv, marker_styles, get_all_profile_files
import argparse

parser = argparse.ArgumentParser(description='Draw multiple profile data.')
parser.add_argument('--draw_option', default='accuracy',
                    help='accuracy or latency')
parser.add_argument('--resnet_layer', type=int, default=34,
                    help='ResNet layer number')
args = parser.parse_args()

option = args.draw_option
resnet_layer = args.resnet_layer

csv_files = get_all_profile_files(
    f"profile_data_resnet{resnet_layer}_noise_0.*.csv")
file_num = len(csv_files)

# Get colors from 'husl' palette
colors = sns.color_palette('husl', n_colors=file_num)
fig, ax1 = plt.subplots()

for i in range(file_num):
    df = read_df_from_csv(csv_files[i])
    if option == 'accuracy':
        sns.lineplot(data=df, x="Pruning_ratio", y="Accuracy",
                     marker='o', ax=ax1,  color=colors[i], label='Noise: 0.'+str(i), markers=marker_styles[i])
    elif option == 'latency':
        sns.lineplot(data=df, x="Pruning_ratio", y="Latency",
                     marker='o', ax=ax1,  color=colors[i], label='Noise: 0.'+str(i), markers=marker_styles[i])
    else:
        raise ValueError('Invalid draw option.')

ax1.set_xlabel('Pruning Ratio')
ax1.set_xlim([-0.05, 1.0])
if option == 'accuracy':
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim([20, 100])
elif option == 'latency':
    ax1.set_ylabel('Latency (ms)')

else:
    raise ValueError('Invalid draw option.')

ax1.grid(True, linestyle='--')  # Add dashed grid lines
plt.legend()
plt.show()
