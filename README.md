[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8333933.svg)](https://doi.org/10.5281/zenodo.8333933)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python)](https://www.python.org/) 
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.3-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.13/) 
[![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

## Overview

The quality of transcriptions of heritage documents produced by optical character recognition (OCR, for printed documents), handwritten text recognition (HTR, for manuscripts) or automatic speech recognition  (ASR, for audio documents) has a major impact on the ability to search and process historical texts. This is the implementation of [Post-correction of Historical Text Transcripts with Large Language Models: An Exploratory Study](https://infoscience.epfl.ch/record/307961).

#### Can LLMs amend poor historical transcriptions?

* **Ability to correct.** Do LLMs improve, degrade, or leave the input text intact?
* **Sensitivity to variations in input text and instructions.** Does LLM post-correction performance depend on the noise of the original document? How sensitive is it to prompt instructions?
* **Real-world applicability.** How do open-access models compare with the limited-access ones? Could millions of noisy historical documents be easily corrected?

## Repository organisation

`lib`: main codebase for the experiments. Details [here]().
`notebooks`: Jupyter notebooks for data error analysis. Details [here]().
`data`: the data samples utilised in our experimental setup. Details [here]().

### Requirements
`>= python 3.9`

Here is a from-scratch script.
```bash

# install dependencies
pip install -r lib/requirements.txt

pip install git+https://github.com/huggingface/transformers
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



