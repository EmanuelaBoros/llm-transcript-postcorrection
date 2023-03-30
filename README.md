# gpt-text-correction
A repository for prelimary work on HTR/OCR/ASR post-correction based on GPT models.


## Overview
#### In general:
_Are GPT (contemporary) language models fit for historical texts?_

#### In particular:
_Can GPT (contemporary) language models correct automatically transcribed text from cultural heritage collections?_

The objective of this project is to test the capacity of GPT (and not only) language models to correct text.

## Setup

### Requirements
`>= python 3.9`

Here is a from-scratch script.
```bash

# install dependencies
pip install -r lib/requirements.txt
```
Costs GPT can be seen at our [OpenAI group](https://platform.openai.com/account/usage).

## Inference

```python
python main.py --input_dir ../data/datasets \
               --output_dir ../data/outputs \
               --config_file ../data/config.yml \
               --prompt_dir ../data/prompts \
               --device cpu
```

### Prompts
#### Zero-shot vs. 1-2-3 shot

### Models
- gpt-3.5-turbo
- gpt-4
- text-curie-001
- text-davinci-003
- davinci
- text-davinci-002
- facebook/opt-350m: # OPT is very inconsistent
- google/flan-t5-base: # Tensorflow issue
- EleutherAI/gpt-neox-20b
- bigscience/bloom
- cerebras/Cerebras-GPT-1.3B


