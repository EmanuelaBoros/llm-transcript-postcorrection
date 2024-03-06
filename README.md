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

* `lib`: main codebase for the experiments. Details [here]().
* `notebooks`: Jupyter notebooks for data error analysis. Details [here]().
* `data`: the data samples utilised in our experimental setup. Details [here]().

### Requirements
`>= python 3.9`

```bash
# install dependencies
pip install -r lib/requirements.txt
# install transformers from the repository
pip install git+https://github.com/huggingface/transformers
```

### Models
| Model     | Release Date | Sizes         | Access  | Max Length |
|-----------|--------------|---------------|---------|------------|
| GPT-2     | 11.2019      | 1.5B          | Open    | 1,024      |
| GPT-3     | 06.2020      | 175B          | Limited | 2,049      |
| GPT-3.5   | 03.2023      | Unknown       | Limited | 4,096      |
| GPT-4     | 03.2023      | Unknown       | Limited | 8,192      |
| BLOOM     | 07.2022      | 560M, 3B, 7.1B | Open    | 2,048      |
| BLOOMZ    | 11.2022      | 560M, 3B, 7.1B | Open    | 2,048      |
| OPT       | 05.2022      | 350M, 6.7B    | Open    | 2,048      |
| LLaMA     | 02.2023      | 7B            | Open    | 2,048      |
| LLaMA-2   | 07.2023      | 7B            | Open    | 4,096      |


