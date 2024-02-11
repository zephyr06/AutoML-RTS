import pytest
from RecordIO.WritingInfo import WritingInfo, get_output_file_name


@pytest.mark.timeout(1)
def test_writing_info():
    writing_info = WritingInfo(pruning_ratio=0.1, model_name='resnet18',  accuracy=0.2,
                               latency=0.3, testing_data_noise=0.4, training_poison_chance=0.1)
    assert "profile_data_resnet18_noise_0.4_poison_0.1.csv" == get_output_file_name(
        writing_info.model_name, testing_data_noise=0.4, training_poison_chance=0.1)
