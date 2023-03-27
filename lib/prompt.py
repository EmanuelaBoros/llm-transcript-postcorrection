import openai
from transformers import BloomForCausalLM, BloomTokenizerFast

class Prompt:

    def __call__(self, prompt, options=None):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        pass


class GPTPrompt(Prompt):
    def __init__(self, api_key, model='text-davinci-002'):
        openai.api_key = api_key

        self.completion = openai.Completion
        self.options = {
            'engine': model,
            'temperature': 0.25,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        return self.completion.create(prompt=prompt, **options)['choices'][0]['text']

    # def summarize(self, text):
    #     prompt = f'Try to summarize the following text as best you can!\n\n{text}'
    #
    #     return self.prediction(prompt=prompt)

class BLOOMPrompt(Prompt)

    def __init__(self, api_key, model="bigscience/bloom-1b3"):
        self.model = BloomForCausalLM.from_pretrained(model)
        self.tokenizer = BloomTokenizerFast.from_pretrained(model)

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Greedy Search
        results = self.tokenizer.decode(self.model.generate(inputs["input_ids"],
                                              max_length=self.options.max_tokens
                                              )[0])

        # Beam Search
        results = self.tokenizer.decode(self.model.generate(inputs["input_ids"],
                                              max_length=self.options.max_tokens,
                                              num_beams=2,
                                              no_repeat_ngram_size=2,
                                              early_stopping=True
                                              )[0])

        # Sampling Top-k + Top-p
        results = self.tokenizer.decode(self.model.generate(inputs["input_ids"],
                                              max_length=self.options.max_tokens,
                                              do_sample=True,
                                              top_k=50,
                                              top_p=0.9
                                              )[0])

        return results