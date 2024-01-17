import pytest
from RecordIO.WritingInfo import WritingInfo, get_output_file_name

def test_writing_info():
    writing_info = WritingInfo(pruning_ratio=0.1, model_name='resnet18',  accuracy=0.2, latency=0.3, training_data_noise=0.4)
    assert "profile_data_resnet18_noise_0.4.csv" == get_output_file_name(writing_info.model_name, writing_info.training_data_noise)
