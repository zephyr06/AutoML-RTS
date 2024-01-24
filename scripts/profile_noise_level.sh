#!/bin/bash

for noise_level in $(seq 0.0 0.1 0.9); do
    python profile_torch_pruner.py --training_noise $noise_level
done
