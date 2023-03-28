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
from data import NERDataset
# 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

logger = logging.getLogger("gpt3-experiments")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODELS = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "davinci", "bigscience/bloom-1b3"]
# CHAT_MODELS = ["gpt-3.5-turbo"] # optimized for chat


def gpt3_query(headers, data, model) -> str:
    response = openai.Completion.create(model=model, prompt=data['prompt'],
                                        temperature=data['temperature'],
                                        max_tokens=data['max_tokens'], top_p=1.0,
                                        frequency_penalty=0.0,
                                        presence_penalty=0.0)
    # import pdb;pdb.set_trace()
    return response["choices"][0]['text']
    # return r_json["choices"][0]["text"]


def prompt_md(prompt: str, gen_text: str) -> str:
    lines = prompt.split("\n")
    prompt_bold = "\n".join([f"**{line}**" if line != "" else line for line in lines])

    return f"{prompt_bold}{gen_text}"



def generate(
    prompt: str = "prompt.txt",
    config_file: str = "config.yml",
    markdown: bool = True,
    query_async: bool = False,
) -> None:
    """
    Generates texts via GPT-3 and saves them to a file.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)

    # If prompt is a file path, load the file as the prompt.
    if os.path.exists(prompt):
        logger.info(f"Loading prompt from {prompt}.")
        with open(prompt, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        logger.info(f"GPT-3 Model Prompt: {prompt}.")

    extension = "md" if markdown else "txt"
    sample_delim = "\n---\n" if markdown else ("=" * 20)

    openai.api_key = c['SECRET_KEY']

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {c['SECRET_KEY']}",
    }

    with open('sentences.txt', 'r') as f:
        sentences = f.readlines()

    for text in sentences:
        data = {
            "prompt": ''.join([prompt, text]),
            "max_tokens": c["max_tokens"],
        }

        loop = asyncio.get_event_loop()

        for temp in c["temperatures"]:
            data.update({"temperature": temp})

            n = c["num_generate"] if temp != 0.0 else 1
            n_str = "samples" if n > 1 else "sample"
            output_file = f"output_{str(temp).replace('.', '_')}.{extension}"
            logger.info(f"Writing {n} {n_str} at temperature {temp} to {output_file}.")


            gen_texts = []
            for _ in trange(n):
                gen_texts.append(gpt3_query(headers, data))
                time.sleep(30)

            with open(output_file, "w", encoding="utf-8") as f:
                for gen_text in gen_texts:
                    if gen_text:
                        gen_text = prompt_md(prompt, gen_text) if markdown else gen_text
                        f.write("{}\n{}\n".format(gen_text, sample_delim))

        loop.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    # parser.add_argument(
    #     "--base_wikidata",
    #     type=str,
    #     help="Base folder with Wikidata data.",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     help="Print lots of debugging statements",
    #     action="store_const",
    #     dest="loglevel",
    #     const=logging.DEBUG,
    #     default=logging.WARNING,
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     help="Be verbose",
    #     action="store_const",
    #     dest="loglevel",
    #     const=logging.INFO,
    # )

    args, _ = parser.parse_known_args()
    # fire.Fire(gpt3_generate)
    dataset = NERDataset(args.input_file)
    # import pdb;pdb.set_trace()