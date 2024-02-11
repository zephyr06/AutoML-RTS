#!/bin/bash

for noise_level in $(seq 0.0 0.1 0.9); do
    python scripts/profile_torch_pruner.py --training_poison_chance 0 --testing_data_noise $noise_level
done
