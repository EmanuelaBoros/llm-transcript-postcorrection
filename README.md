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
#### Zero-shot 

*Example gpt-4:*
```json
{"prompt": "Correct the spelling and grammar of the following incorrect text from on optical character recognition (OCR) applied to a historical document:\n\nIncorrect text: The European Commi66ion said on Thursday it disagreed with German advice to consumers to shun Brifish ss ..ff lamb until scientists determine whether mad cow disease can be transmitted to sheep.\nThe corrected text is:", 
"max_tokens": 512, 
"ocr_text": "The European Commi66ion said on Thursday it disagreed with German advice to consumers to shun Brifish ss ..ff lamb until scientists determine whether mad cow disease can be transmitted to sheep.",
"correct_text": "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep.", 
"temperature": 0.1, 
"prediction": "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep.", "num_generate": 0}
```
*Example OPT-350M* (small model):
```json
{"prompt": "Correct the spelling of the following text from on optical character recognition (OCR) applied to a historical document:\n\nText: The European Commi66ion said on Thursday it disagreed with German advice to consumers to shun Brifish ss ..ff lamb until scientists determine whether mad cow disease can be transmitted to sheep.\nCorrected text:", 
"max_tokens": 512, 
"ocr_text": "The European Commi66ion said on Thursday it disagreed with German advice to consumers to shun Brifish ss ..ff lamb until scientists determine whether mad cow disease can be transmitted to sheep.",
"correct_text": "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep.", 
"prediction": "The European Commi66ion said on Thursday it disagreed with German advice to consumers to shun Brifish ss..ff lamb until scientists determine whether mad cow disease can be transmitted to sheep.\n\nThe new variant has been detected in a flock of 5 sheep in the north of England, although the exact origin of the strain is not known.\n\nAccording to the news service Reuters, officials at the National Health Service (NHS) will conduct a field test of the new variant to be able to determine its potential to spread through livestock.\n\nThere are fears that it is the most lethal type of cow virus known and is likely to be spread through the supply chain between sheep and pigs.\n\nThe new strain is the second in a series of attacks on beef animals since April. The first attack on lamb meat was in March when the strain was first detected in sheep in southern France.\n\nThe new mutation of cow virus caused a severe stomach upset in sheep in the United Kingdom and also caused stomach cancer in sheep in the north of France.</s>", "num_generate": 2}

```

#### 1-2-3 shot

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


