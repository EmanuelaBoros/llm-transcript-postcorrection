#!/bin/bash

# List of datasets
datasets=("impresso" "ajmc" "icdar" "overproof")

# List of prompts
prompts=("prompt_basic_01.txt", "prompt_basic_02.txt", "prompt_complex_01.txt")

# Export the run.sh script path
export SCRIPT_PATH=$(pwd)/run.sh

# Run experiments in parallel
parallel --jobs 0 "$SCRIPT_PATH" ::: "${datasets[@]}" ::: "${prompts[@]}"

echo "All training runs completed."
