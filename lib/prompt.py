import openai
from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Prompt:

    def __call__(self, prompt, options=None, device='cpu'):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        pass


class GPTPrompt(Prompt):
    '''
    Prompting for GPT-2 anf GPT-3 through their API
    '''
    def __init__(self, api_key=None, model='text-davinci-003', device='cpu'):
        openai.api_key = api_key

        self.options = {
            'engine': model,
            'temperature': 0.7,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        try:
            result = openai.Completion.create(
                prompt=prompt, **options)['choices'][0]['text']
        except:
            result = openai.ChatCompletion.create(
                **options,
                messages=[{"role": "user", "content": prompt}]
            )

        return result



class HFPrompt(Prompt):
    '''
    Prompting for HuggingFace models that can be loaded with AutoForCausalLM
    '''
    def __init__(self, api_key=None, model="bigscience/bloom", device='cpu'):
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def prediction(self, prompt, options=None, search='topk'):

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if search == 'greedy':
            # Greedy Search
            result = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    temperature=options['temperature'],
                    max_length=options['max_tokens'])[0])

        elif search == 'beam':
            # Beam Search
            result = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=options['max_tokens'],
                    num_beams=2,
                    temperature=options['temperature'],
                    no_repeat_ngram_size=2,
                    early_stopping=True)[0])

        else:
            # in case no specific request, apply the best one: top-k
            # Sampling Top-k + Top-p
            result = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=options['max_tokens'],
                    do_sample=True,
                    temperature=options['temperature'],
                    top_k=50,
                    top_p=0.9)[0])

        # OPT adds the prompt in the response, so we are removing it
        last_comment_in_prompt = prompt.split('\n')[-1]
        result = result[result.index(
            last_comment_in_prompt) + len(last_comment_in_prompt) + 1:]

        return result
