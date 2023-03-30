import openai
from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Prompt:

    def __call__(self, prompt, options=None):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        pass


class GPTPrompt(Prompt):
    def __init__(self, api_key=None, model='text-davinci-003'):
        openai.api_key = api_key

        self.completion = openai.Completion
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

        return self.completion.create(
            prompt=prompt, **options)['choices'][0]['text']


class OPTPrompt(Prompt):

    def __init__(self, api_key=None, model="facebook/opt-350m"):

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = OPTForCausalLM.from_pretrained(model)  # .to('cuda')
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

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # # Greedy Search
        # results = self.tokenizer.decode(
        #     self.model.generate(
        #         inputs["input_ids"],
        #         max_length=self.options['max_tokens'])[0])
        #
        # import pdb;
        # pdb.set_trace()
        #
        # # Beam Search
        # results = self.tokenizer.decode(
        #     self.model.generate(
        #         inputs["input_ids"],
        #         max_length=self.options['max_tokens'],
        #         num_beams=2,
        #         no_repeat_ngram_size=2,
        #         early_stopping=True)[0])

        # Sampling Top-k + Top-p
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options['max_tokens'],
                do_sample=True,
                top_k=50,
                top_p=0.9)[0])

        return results


class FlanT5Prompt(Prompt):

    def __init__(self, api_key=None, model="google/flan-t5-xxl"):

        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, device_map="auto")

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Greedy Search
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens)[0])

        # Beam Search
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens,
                num_beams=2,
                no_repeat_ngram_size=2,
                early_stopping=True)[0])

        # Sampling Top-k + Top-p
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.9)[0])

        return results


class BLOOMPrompt(Prompt):

    def __init__(self, api_key=None, model="bigscience/bloom-1b3"):
        self.model = BloomForCausalLM.from_pretrained(model)
        self.tokenizer = BloomTokenizerFast.from_pretrained(model)

    def prediction(self, prompt, options=None):
        import pdb
        pdb.set_trace()

        if not options:
            options = self.options

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Greedy Search
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens)[0])

        # Beam Search
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens,
                num_beams=2,
                no_repeat_ngram_size=2,
                early_stopping=True)[0])

        # Sampling Top-k + Top-p
        results = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_length=self.options.max_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.9)[0])

        return results


class GPTNeoxPrompt(Prompt):

    def __init__(self, api_key=None, model="EleutherAI/gpt-neox-20b"):

        self.model = GPTNeoXForCausalLM.from_pretrained(model)
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model)

        # prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."
        #
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        #
        # gen_tokens = model.generate(
        #     input_ids,
        #     do_sample=True,
        #     temperature=0.9,
        #     max_length=100,
        # )
        # gen_text = tokenizer.batch_decode(gen_tokens)[0]

        def prediction(self, prompt, options=None):
            if not options:
                options = self.options

            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Greedy Search
            results = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=self.options.max_tokens)[0])

            # Beam Search
            results = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=self.options.max_tokens,
                    num_beams=2,
                    no_repeat_ngram_size=2,
                    early_stopping=True)[0])

            # Sampling Top-k + Top-p
            results = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=self.options.max_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9)[0])

            return results
