import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from ResNet.variables import ROOT_PATH
import seaborn as sns
from utils import read_df_from_csv, marker_styles, get_all_profile_files


csv_files = get_all_profile_files("profile_data_resnet18_noise_0.*.csv")
file_num = len(csv_files)

# Get colors from 'husl' palette
colors = sns.color_palette('husl', n_colors=file_num)
fig, ax1 = plt.subplots()

for i in range(file_num):
    df = read_df_from_csv(csv_files[i])
    sns.lineplot(data=df, x="Pruning_ratio", y="Accuracy",
                 marker='o', ax=ax1,  color=colors[i], label='Noise: 0.'+str(i), markers=marker_styles[i])


ax1.set_xlabel('Pruning Ratio')
ax1.set_ylim([20, 100])
ax1.set_xlim([-0.05, 1.0])
ax1.set_ylabel('Accuracy (%)')

ax1.grid(True, linestyle='--')  # Add dashed grid lines
plt.legend()
plt.show()
