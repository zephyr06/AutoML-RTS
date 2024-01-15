import csv


def create_csv_file(file_name, variable_names):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(variable_names)


def save_to_file(input_data, output_data, file_path):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the data row
        writer.writerow(list(input_data) + list(output_data))


# TODO: consider how to read the file and use it;
def read_from_file(file_path):
    input_output_pairs = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row and skip it
        for row in reader:
            input_data, output_data, data_type = row
            input_output_pairs.append((input_data, output_data, data_type))
    return input_output_pairs
