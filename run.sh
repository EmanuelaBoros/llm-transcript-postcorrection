#!/bin/bash

# List of datasets
datasets=("impresso" "ajmc" "icdar" "overproof")

# List of prompts
prompts=("prompt_basic_01.txt", "prompt_basic_02.txt", "prompt_complex_01.txt")

# Base directories and options
input_dir="../data/datasets/ocr/converted/"
output_dir="../data/output/"
device="cuda"

# Loop through each dataset and run the training algorithm
for dataset in "${datasets[@]}"; do
    for prompt in "${prompts[@]}"; do
        echo "Training on dataset: $dataset with prompt: $prompt"
        python main.py --input_dir "${input_dir}${dataset}" --output_dir "${output_dir}${dataset}_${prompt%.*}" --device "$device" --prompts "../data/prompts/" --prompt "$prompt"
        echo "Training complete for dataset: $dataset with prompt: $prompt"
    done
done

echo "All training runs completed."
