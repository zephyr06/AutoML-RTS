import pandas as pd
import matplotlib.pyplot as plt
import glob
from ResNet.variables import ROOT_PATH
import os
import seaborn as sns

sns.set_palette("husl")


def draw_profile_data_2metric(csv_file_name):
    # Find all CSV files that match the pattern
    csv_file_name = os.path.join(ROOT_PATH, "profile_data", csv_file_name)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_name, names=[
                     "Pruning_ratio", "Accuracy", "Latency"], header=0)

    # Get colors from 'husl' palette
    colors = sns.color_palette('husl', n_colors=2)

    # Plot the data
    fig, ax1 = plt.subplots()

    # Plot the first metric on the first y-axis (Accuracy)
    sns.lineplot(data=df, x="Pruning_ratio", y="Accuracy",
                 marker='o', ax=ax1, label='Accuracy', color=colors[0])
    ax1.set_xlabel('Pruning Ratio')
    ax1.set_ylabel('Accuracy (%)', color=colors[0])
    ax1.tick_params(axis='y')
    ax1.set_ylim([20, 105])

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    # Plot the second metric on the second y-axis (Latency)
    sns.lineplot(data=df, x="Pruning_ratio", y="Latency",
                 marker='*', ax=ax2, label='Latency', color=colors[1])
    ax2.set_ylabel('Latency (s)', color=colors[1])
    ax2.tick_params(axis='y')
    ax2.set_ylim([0.8, 1.4])

    ax1.set_xlim([-0.1, 1.0])

    ax1.grid(linestyle='--')
    ax2.grid(linestyle='--')
    # Add title and labels
    plt.title('Profile Test Fast Resnet18')
    plt.xlabel('Pruning Ratio')

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    draw_profile_data_2metric(
        "profile_data_resnet18_noise_0.2.csv")
