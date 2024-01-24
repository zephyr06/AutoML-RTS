
class WritingInfo:
    def __init__(self, pruning_ratio=-1, accuracy=-1, latency=-1, training_data_noise=0, model_name="", profile_csv_file_name: str = None):
        self.pruning_ratio = pruning_ratio
        self.accuracy = accuracy
        self.latency = latency

        if profile_csv_file_name:
            parts = profile_csv_file_name.split('_')
            self.model_name = parts[2]
            self.training_data_noise = parts[4]
        else:
            self.training_data_noise = training_data_noise
            self.model_name = model_name



    
def get_output_file_name(model_name, training_data_noise):
    file_name = f"profile_data_{model_name}_noise_{training_data_noise}.csv"
    return file_name

    