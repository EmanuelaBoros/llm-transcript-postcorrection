[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8333933.svg)](https://doi.org/10.5281/zenodo.8333933)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python)](https://www.python.org/) 
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.3-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.13/) 
[![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

The quality of automatic transcription of heritage documents, whether from printed, manuscripts or audio sources, has a decisive impact on the ability to search and process historical texts. Although significant progress has been made in text recognition (OCR, HTR, ASR), textual materials derived from library and archive collections remain largely erroneous and noisy. Effective post-transcription correction methods are therefore necessary and have been intensively researched for many years. As large language models (LLMs) have recently shown exceptional performances in a variety of text-related tasks, we investigate their ability to amend poor historical transcriptions. We evaluate fourteen foundation language models against various post-correction benchmarks comprising different languages, time periods and document types, as well as different transcription quality and origins. We compare the performance of different model sizes and different prompts of increasing complexity in zero and few-shot settings. Our evaluation shows that LLMs are anything but efficient at this task. Quantitative and qualitative analyses of results allow us to share valuable insights for future work on post-correcting historical texts with LLMs.

## Overview

This is the repository for the work on HTR/OCR/ASR post-correction based on LLMs.

#### In general:
_Are GPT (contemporary) language models fit for historical texts?_

#### In particular:
_Can GPT (contemporary) language models correct automatically transcribed text from cultural heritage collections?_

The objective of this project is to test the capacity of GPT (and not only) language models to correct text.

## Organization

`data/config.yml`- configuration file for all LLMs and theire specificities
Example for GPT-4 and and example prompt `prompt_basic_01.txt`:
```
max_tokens: 512

# Lists for loops
models:
  - gpt-4:
      - class: GPTPrompt
      - prompt: prompt_basic_01.txt
      - num_generate: 1
      - temperatures:
          - 0.1
          - 0.5
          - 0.9
          - 1.0
          - 1.5
          - 1.9
          - 2.0
  ```

`data/datasets`- folder with the datasets (for now, just a simulated datatset `test`)

`data/prompts/`- folder with the prompts in format `txt`
Example `prompt_basic_01.txt`:
```
Correct the text: {{TEXT}}
```

`data/outputs/`- folder with the results per prompt type

### Requirements
`>= python 3.9`

Here is a from-scratch script.
```bash

# install dependencies
pip install -r lib/requirements.txt

For LLama:
Install 🤗 Transformers from source with the following command:

pip install git+https://github.com/huggingface/transformers
```
Costs GPT can be seen at our [OpenAI group](https://platform.openai.com/account/usage).

## Inference

```python
python main.py --input_dir ../data/datasets/converted \
               --output_dir ../data/outputs \
               --config_file ../data/config.yml \
               --prompt_dir ../data/prompts \
               --device cpu
```


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

#### Fine-tuning
`export OPENAI_API_KEY="<OPENAI_API_KEY>"`
```
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

`openai tools fine_tunes.prepare_data -f <LOCAL_FILE>`
`openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>`
Every fine-tuning job starts from a base model, which defaults to curie. The choice of model influences both the performance of the model and the cost of running your fine-tuned model. Your model can be one of: ada, babbage, curie, or davinci. Visit our pricing page for details on fine-tune rates.
`openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>`
```
# List all created fine-tunes
openai api fine_tunes.list

# Retrieve the state of a fine-tune. The resulting object includes
# job status (which can be one of pending, running, succeeded, or failed)
# and other information
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# Cancel a job
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>
```

Usage of the fine-tuned model:
`openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>`
```
import openai
openai.Completion.create(
    model=FINE_TUNED_MODEL,
    prompt=YOUR_PROMPT)
```
Delete the model:
`openai api models.delete -i <FINE_TUNED_MODEL>`
  

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

### How To's

#### Run a zero-shot scenario

## For the zero-shot scenario with the English prompts that increase in difficulty:

```
PROMPT in [prompt_basic_01.txt, prompt_basic_02.txt, prompt_complex_01.txt, prompt_complex_02.txt, prompt_complex_03_en.txt]
```
For the datasets [icdar-2017, icdar-2019, impresso-nzz, overproof], the ``PROMPT = prompt_complex_03.txt``
```
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python main.py --input_dir ../data/datasets/ocr/converted/ajmc-mixed/ --output_dir ../data/output/ --config_file ../data/config.yml --prompt_dir ../data/prompts --device cuda --prompt prompt_basic_01.txt
```

for ``PROMP of complexity 03``:
For the datasets [ajmc-mixed, ajmc-primary, htrec, ina]:
```
PROMPT in [prompt_complex_03_ajmc_mixed_en.txt, prompt_complex_03_ajmc_primary_en.txt, prompt_complex_03_htrec_en.txt, prompt_complex_03_ina_en.txt]
```

## For the zero-shot scenario with the language-specific prompts:
For the datasets [icdar-2017, icdar-2019, impresso-nzz, overproof], the ``PROMPT = prompt_complex_03.txt``
```
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python main.py --input_dir ../data/datasets/ocr/converted/icdar-2017/ --output_dir ../data/output/ --config_file ../data/config.yml --prompt_dir ../data/prompts --device cuda --prompt prompt_complex_03.txt
```

for ``PROMP of complexity 03``:
For the datasets [ajmc-mixed, ajmc-primary, htrec, ina]:
```
PROMPT in [prompt_complex_03_ajmc_mixed_el.txt, prompt_complex_03_ajmc_primary_el.txt, prompt_complex_03_htrec_el.txt, prompt_complex_03_ina_fr.txt]
```

## For the few-shot scenario with the English prompts that increase in difficulty:

## For the few-shot scenario with the language-specific prompts:



