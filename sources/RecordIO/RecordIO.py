import csv
import os
import numpy as np
import pandas as pd

from ResNetTrain.variables import ROOT_PATH
from RecordIO.WritingInfo import WritingInfo, get_output_file_name


def get_csv_file(file_name):
    file_path = os.path.join(ROOT_PATH, "profile_data",
                             file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        with open(file_path, 'w') as file:
            file.write("Pruning_ratio, Accuracy, Latency\n")
    return file_path


def save_to_file(input_data, output_data, file_path):
    with open(file_path, 'a') as file:
        row = ', '.join(map(str, input_data+output_data))+'\n'
        file.write(row)


def whether_contain_header(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        first_line = lines[0]
        if (first_line[0].isdigit()):
            return False
        else:
            return True


def query_profiled_result(prune_ratio, profile_csv_file_name=None):
    """Query the profiled result from the specific profile_csv_file, return accuracy and latency"""
    if profile_csv_file_name:
        path = os.path.join(ROOT_PATH, "profile_data", profile_csv_file_name)
        if whether_contain_header(path):
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=None)
        if not df.empty:
            df.columns = ['prune_ratio', 'accuracy', 'latency']
            row = df[np.isclose(df.iloc[:, 0], prune_ratio, rtol=0.01)]
            if not row.empty:
                return row.iloc[0]['accuracy'], row.iloc[0]['latency']
    return None, None


def find_record(model_name, training_data_noise, pruning_ratio):
    file_name = get_output_file_name(model_name, training_data_noise)
    file_path = get_csv_file(file_name)
    if not os.path.exists(file_path):
        return None, None

    return query_profiled_result(pruning_ratio, file_name)


def get_date():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
