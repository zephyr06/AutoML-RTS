import pandas as pd
import matplotlib.pyplot as plt
import glob
from ResNet.variables import ROOT_PATH
import os


def draw_profile_data_2metric(csv_file_name):
    # Find all CSV files that match the pattern
    csv_file_name = os.path.join(ROOT_PATH, "profile_data", csv_file_name)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_name, header=None)

    # Plot the data
    # plt.plot(df[0], df[1], label="accuracy (%)")
    # plt.plot(df[0], df[2], label="Latency (s)")
    fig, ax1 = plt.subplots()

    # Plot the first metric on the first y-axis
    ax1.plot(df[0], df[1], color='tab:blue', marker='o')
    ax1.set_xlabel('Pruning Ratio')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    # Plot the second metric on the second y-axis
    ax2.plot(df[0], df[2], color='tab:red', marker='*')
    ax2.set_ylabel('Latency (s)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.set_xlim([-0.1, 1.0])

    ax1.grid(linestyle='--')
    ax2.grid(linestyle='--')
    # Add title and labels
    plt.title('Profile Test Fast Resnet18')
    plt.xlabel('Pruning Ratio')
    # plt.ylabel('Accuracy')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    draw_profile_data_2metric(
        "profile_test_fast_resnet18_pruning_ratio2024-01-15-21-22-34.csv")
