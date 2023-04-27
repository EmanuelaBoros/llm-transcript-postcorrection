import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


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
        self.device = device
        self.options = {
            'engine': model,
            'temperature': 0.7,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        # max_tokens = len(self.tokenizer(prompt, return_tensors="pt").to(self.device))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        options['max_tokens'] = inputs['input_ids'].shape[1] * 2 + 1
        # print(options['max_tokens'])
        if 'davinci' in options['engine']:
            max_model_tokens = 2049
        else:
            max_model_tokens = 1024
        if options['max_tokens'] > max_model_tokens:
            inputs['input_ids'] = inputs['input_ids'][:, :max_model_tokens//2]
            options['max_tokens'] = max_model_tokens

            prompt = self.tokenizer.decode(*inputs['input_ids'])
        import pdb;pdb.set_trace()
        try:
            result = openai.Completion.create(prompt=prompt, **options)['choices'][0]['text']
        except BaseException as ex:
            print(f'Error: {ex}')
            # {
            #     "model": "text-davinci-edit-001",
            #     "input": "What day of the wek is it?",
            #     "instruction": "Fix the spelling mistakes",
            # }
            # Chat endpoints do not have the same engine

            options.update({'model': options['engine']})
            options.pop('engine')
            result = openai.ChatCompletion.create(
                **options, messages=[{"role": "user", "content": prompt}])['choices'][0]['message']['content']

        return result


class HFPrompt(Prompt):
    '''
    Prompting for HuggingFace models that can be loaded with AutoForCausalLM
    '''

    def __init__(self, api_key=None, model="bigscience/bloom", device='cpu'):

        self.device = device
        if 'llama' in model.lower():

            self.tokenizer = LlamaTokenizer.from_pretrained(model)
            self.model = LlamaForCausalLM.from_pretrained(model)

        elif "alpaca" in model.lower():
            from peft import PeftModel

            self.tokenizer = LlamaTokenizer.from_pretrained(
                "decapoda-research/llama-7b-hf")

            BASE_MODEL = "decapoda-research/llama-7b-hf"
            LORA_WEIGHTS = "tloen/alpaca-lora-7b"

            if device == "cuda":
                model = LlamaForCausalLM.from_pretrained(
                    BASE_MODEL,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.model = PeftModel.from_pretrained(
                    model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True)
            elif device == "mps":
                model = LlamaForCausalLM.from_pretrained(
                    BASE_MODEL,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
                self.model = PeftModel.from_pretrained(
                    model,
                    LORA_WEIGHTS,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
                )
                self.model = PeftModel.from_pretrained(
                    model,
                    LORA_WEIGHTS,
                    device_map={"": device},
                )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def prediction(self, prompt, options=None, search='topk'):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        options['max_tokens'] = inputs['input_ids'].shape[1] * 2 + 1
        # print(options['max_tokens'])
        max_model_tokens = self.model.config.max_position_embeddings
        if options['max_tokens'] > max_model_tokens:
            inputs['input_ids'] = inputs['input_ids'][:, :max_model_tokens//2]
            options['max_tokens'] = max_model_tokens

        # print(options['max_tokens'])
        # print('--'*100)
        # # import pdb;
        # # pdb.set_trace()

        if search == 'greedy':
            # Greedy Search
            result = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=options['temperature'],
                    max_length=options['max_tokens']
                )[0])

        elif search == 'beam':
            # Beam Search
            result = self.tokenizer.decode(
                self.model.generate(
                    inputs["input_ids"],
                    max_length=options['max_tokens'],
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=2,
                    no_repeat_ngram_size=2,
                    early_stopping=True)[0])
        else:
            # in case no specific request, apply the best one: top-k
            # Sampling Top-k + Top-p

            # with torch.no_grad():
            #     generation_output = model.generate(
            #         input_ids=input_ids,
            #         generation_config=generation_config,
            #         return_dict_in_generate=True,
            #         output_scores=True,
            #         max_new_tokens=max_new_tokens,
            #     )

            result = self.tokenizer.decode(self.model.generate(input_ids=inputs["input_ids"], max_length=options['max_tokens'], pad_token_id=self.tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.9)[0])

        # OPT adds the prompt in the response, so we are removing it
        last_comment_in_prompt = prompt.split('\n')[-1]
        if last_comment_in_prompt in result:
            result = result[result.index(
                last_comment_in_prompt) + len(last_comment_in_prompt) + 1:]

        return result
