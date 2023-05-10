#!/bin/bash

# Base directories and options
input_dir="../data/datasets/ocr/converted/"
output_dir="../data/output/"
device="cuda"

# Get dataset and prompt from arguments
dataset="$1"
prompt="$2"

# Run the training algorithm
echo "Training on dataset: $dataset with prompt: $prompt"
python main.py --input_dir "${input_dir}${dataset}" --output_dir "${output_dir}${dataset}_${prompt%.*}" --device "$device" --prompts "../data/prompts/" --prompt "$prompt"
echo "Training complete for dataset: $dataset with prompt: $prompt"
