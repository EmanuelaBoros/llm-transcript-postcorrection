import openai
import yaml
import os
import asyncio
import fire
import argparse
from tqdm import tqdm
import importlib
import jsonlines
import logging
from const import Const
logger = logging.getLogger("gpt-experiments")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_dict(list_of_dicts: list):
    return_list = {}
    for list_entry in list_of_dicts:
        return_list |= list_entry
    return return_list


def generate(
    input_dir: str = "../data/datasets",
    output_dir: str = "../data/output",
    prompt_dir: str = "../data/prompts",
    config_file: str = "../data/config.yml",
    device: str = 'cpu'
) -> None:
    """
    Generates texts via several models ni config and saves them to predictions files.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    openai.api_key = os.getenv("OPENAI_API_KEY") #config['SECRET_KEY']
    if 'prompt' in config:
        prompt_path = os.path.join(prompt_dir, config['prompt'])
        # If prompt is a file path, load the file as the prompt.
        if os.path.exists(prompt_path):
            logger.info(f"Loading prompt from {prompt_path}.")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
        else:
            logger.info(f"Model prompt missing: {prompt_path}.")

    for model in config['models']:

        (model_name, experiment_details), = model.items()

        logger.info('Experimenting with {}'.format(model_name))

        experiment_details = get_dict(experiment_details)

        model_class = experiment_details['class']
        # prompt_path = os.path.join(prompt_dir, experiment_details['prompt'])

        results_dir = os.path.join(
            output_dir,
            config['prompt'].replace(
                '.txt',
                ''))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        module = importlib.import_module('prompt')
        class_ = getattr(module, model_class)
        instance = class_(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            device=device)

        # Iterate in the data folder with all datasets
        print(input_dir)
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                if 'jsonl' in name:
                    input_file = os.path.join(root, name)

                    logging.info(f'Post-correcting {input_file}')
                    dataset_name = name.replace('.jsonl', '')

                    dataset_model_results_dir = os.path.join(results_dir, dataset_name)
                    if not os.path.exists(dataset_model_results_dir):
                        os.makedirs(dataset_model_results_dir)

                    output_file = os.path.join(
                        dataset_model_results_dir, 'results-{}-{}.jsonl'.format(
                            dataset_name, model_name).replace(
                            '/', '-'))

                    logger.info(
                        'Predictions for {} are saved in {}'.format(
                            input_file, output_file))

                    with open(input_file, 'r') as g:
                        total_files = len(g.readlines())

                    progress_bar = tqdm(
                        total=total_files,
                        desc="Processing files",
                        unit="file")

                    already_done = {}
                    with jsonlines.open(output_file, 'w') as f:
                        with jsonlines.open(input_file, 'r') as g:
                            for json_line in g:
                                data = {
                                    Const.PREDICTION: {
                                        }
                                }
                                for TEXT_LEVEL in [Const.LINE, Const.SENTENCE, Const.REGION]:
                                    text = json_line[Const.OCR][TEXT_LEVEL]
                                    if text not in already_done:

                                        if text is not None:
                                            # print(TEXT_LEVEL, text[:20])
                                            data[Const.PREDICTION][Const.PROMPT] = prompt.replace('{{TEXT}}', text)
                                            # logger.info('--Text: {}'.format(data[Const.PREDICTION][Const.PROMPT]))

                                            # loop = asyncio.get_event_loop()
                                            n = experiment_details["num_generate"]
                                            n_str = "samples" if n > 1 else "sample"

                                            options = {
                                                'engine': model_name,
                                                # 'temperature': 0.5,
                                                'top_p': 1.0,
                                                'frequency_penalty': 0,
                                                'presence_penalty': 0
                                                # 'max_tokens': max_tokens
                                            }

                                            result = instance.prediction(
                                                data[Const.PREDICTION][Const.PROMPT], options)

                                            data[Const.PREDICTION][TEXT_LEVEL] = result
                                            # data[Const.PREDICTION].update({"num_generate": idx}) # TODO: for now, just one run

                                            data = json_line | data
                                            f.write(data)
                                            # f.flush()

                                            already_done[text] = result

                                            # loop.close()

                                            # except Exception as ex:
                                            #     logging.warning(f'Exception: {ex} {input_file}')
                                    else:
                                        data[Const.PREDICTION].update({TEXT_LEVEL: already_done[text]})

                                    # data[Const.PREDICTION].update({Const.PROMPT: None})
                                progress_bar.update(1)
                    progress_bar.close()


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
        "--device",
        default='cpu',
        type=str,
        help="The inference is done either on cuda or cpu.",
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
