#!/bin/bash

# Base directories and options
input_dir="data/datasets/ocr/converted/"
output_dir="data/output/"
device="cuda"

# Get dataset and prompt from arguments
dataset="$1"
prompt="$2"
config_file="$3"

# Run the training algorithm
echo "Training on dataset: $dataset with prompt: $prompt"
python lib/main.py --input_dir "${input_dir}${dataset}" --output_dir "${input_dir}${dataset}" --device "$device" --prompts "../data/prompts/" --prompt "$prompt" --config_file "$config_file"
echo "Training complete for dataset: $dataset with prompt: $prompt"
