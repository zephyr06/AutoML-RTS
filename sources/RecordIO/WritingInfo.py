
class WritingInfo:
    def __init__(self, pruning_ratio=0, accuracy=0, latency=0, training_data_noise=0, model_name=""):
        self.pruning_ratio = pruning_ratio
        self.accuracy = accuracy
        self.latency = latency
        self.training_data_noise = training_data_noise
        self.model_name = model_name

    
def get_output_file_name(model_name, training_data_noise):
    file_name = f"profile_data_{model_name}_noise_{training_data_noise}.csv"
    return file_name

    