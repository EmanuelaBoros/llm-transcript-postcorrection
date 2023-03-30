import os
import openai
import yaml
import json
import logging
import os
import asyncio
import fire
import argparse
import time
from tqdm import trange
from dataset import NERDataset
# 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
from prompt import Prompt
import importlib
import jsonlines


logger = logging.getLogger("gpt-experiments")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "davinci",
    "bigscience/bloom-1b3"]
# CHAT_MODELS = ["gpt-3.5-turbo"] # optimized for chat


def gpt3_query(headers, data, model) -> str:
    response = openai.Completion.create(
        model=model,
        prompt=data['prompt'],
        temperature=data['temperature'],
        max_tokens=data['max_tokens'],
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0)
    # import pdb;pdb.set_trace()
    return response["choices"][0]['text']
    # return r_json["choices"][0]["text"]


def prompt_md(prompt: str, gen_text: str) -> str:
    lines = prompt.split("\n")
    prompt_bold = "\n".join(
        [f"**{line}**" if line != "" else line for line in lines])

    return f"{prompt_bold}{gen_text}"


def generate(
    input_dir, output_dir,
    prompt_dir: str = "../data/prompts",
    config_file: str = "../data/config.yml",
    markdown: bool = True,
    query_async: bool = False,
) -> None:
    """
    Generates texts via GPT-3 and saves them to a file.
    """
    with open(os.path.join(input_dir, config_file), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # extension = "md" if markdown else "txt"
    # sample_delim = "\n---\n" if markdown else ("=" * 20)

    openai.api_key = config['SECRET_KEY']

    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {config['SECRET_KEY']}",
    # }

    with open(os.path.join(input_dir, 'test.txt'), 'r') as f:
        sentences = f.readlines()

    for model in config['models']:

        (model_name, details), = model.items()

        model_class = details['class']
        prompt = os.path.join(prompt_dir, details['class'])
        # If prompt is a file path, load the file as the prompt.
        if os.path.exists(prompt):
            logger.info(f"Loading prompt from {prompt}.")
            with open(prompt, "r", encoding="utf-8") as f:
                prompt = f.read()
        else:
            logger.info(f"Model Prompt: {prompt}.")

        module = importlib.import_module('prompt')
        class_ = getattr(module, model_class)
        instance = class_(api_key=config['SECRET_KEY'], model=model_name)
        logger.info('Experimenting with {}'.format(model_name))
        output_file = os.path.join(
            output_dir,
            'results_{}.jsonl'.format(model_name).replace(
                '/',
                '_'))
        with jsonlines.open(output_file, 'w') as f:
            for sentence in sentences:
                data = {
                    "prompt": prompt.replace('{{TEXT}}', sentence),
                    "max_tokens": config["max_tokens"],
                }
                logger.info('--Text: {}'.format(data['prompt']))

                loop = asyncio.get_event_loop()

                for temp in config["temperatures"]:
                    data.update({"temperature": temp})

                    n = config["num_generate"] if temp != 0.0 else 1
                    n_str = "samples" if n > 1 else "sample"
                    # output_file = f"output_{str(temp).replace('.', '_')}.{extension}"
                    logger.info(
                        f"Writing {n} {n_str} at temperature {temp} to {output_file}.")

                    for _ in trange(n):

                        result = instance.prediction(data['prompt'])
                        data.update({"prediction": result})

                    f.write(data)

                loop.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Base folder with input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Base folder for prediction files.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Base folder with prompts data.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()
    fire.Fire(generate)
