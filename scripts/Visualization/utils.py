import pandas as pd


# Define a list of marker styles
marker_styles = ['o', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'H']


def read_df_from_csv(csv_file_name):
    return pd.read_csv(csv_file_name, names=[
        "Pruning_ratio", "Accuracy", "Latency"], header=0)
