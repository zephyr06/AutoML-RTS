import pandas as pd
import os
import glob

from ResNet.variables import ROOT_PATH

# Define a list of marker styles
marker_styles = ['o', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'H']


def read_df_from_csv(csv_file_name):
    return pd.read_csv(csv_file_name, names=[
        "Pruning_ratio", "Accuracy", "Latency"], header=0)


def get_all_profile_files(csv_name_pattern: str):

    data_folder_path = os.path.join(
        ROOT_PATH, "profile_data", csv_name_pattern)
    csv_files = glob.glob(data_folder_path)
    csv_files = sorted(csv_files)
    return csv_files
