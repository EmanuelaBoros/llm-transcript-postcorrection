from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
import glob
from utils import clean_text, align_texts
from const import Const
import os
import json
import argparse
from tqdm import tqdm
import logging
from langdetect import detect
import sys
from qa import qa_bloom, qa_pleias

main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(main_dir)
warnings.filterwarnings("ignore")


def load_metadada(args):
    metadata_path = None
    for path in glob.glob(args.input_dir + "/**/*", recursive=True):
        if "eval_metadata" in path:
            metadata_path = path

    if metadata_path is None:
        print("Metadata was not found.")
        args.metadata = None
    else:
        with open(metadata_path, "r") as f:
            metadata = f.readlines()

        metadata = [line.strip().split(";") for line in metadata]
        # Skip columns line: 'File;Date;Type;NbAlignedChar'
        columns, metadata = metadata[0], metadata[1:]

        metadata = pd.DataFrame(metadata, columns=columns)
        metadata["File"] = metadata["File"].apply(
            lambda x: x.replace("\\", "/")
        )  # Replace Windows style file names

        args.metadata = metadata


def lookup_metadata(args, input_file):
    file_metadata = args.metadata[
        args.metadata.File == "/".join(input_file.split("/")[-2:])
    ]
    return file_metadata.to_dict("records")[0]


def process_file(args, input_file, output_file, dataset_name):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    if args.metadata is not None:
        file_metadata = lookup_metadata(args, input_file)
    else:
        file_metadata = {}

    # Extract OCR and GS sentences from the data list
    # [OCR_toInput] [OCR_aligned] [ GS_aligned]
    gt_text = clean_text(data[2].replace("[ GS_aligned]", "").strip())
    ocr_text = clean_text(data[0].replace("[OCR_toInput]", "").strip())

    language = input_file.split("/")[-3].lower()

    try:
        # Align the OCR and GS sentences
        aligned_sentences = align_texts(gt_text, ocr_text, language=language)
    except BaseException:
        # Defaulting to English
        aligned_sentences = align_texts(gt_text, ocr_text, language="en")

    # Write the output to a JSON Lines file
    with open(output_file, "a") as outfile:
        for gs_sentence, ocr_sentence in aligned_sentences:
            language = input_file.split("/")[-2].lower()
            if "1" in language:
                language = language.replace("1", "")
            if "en" in language:
                language = "en"
            elif "fr" in language:
                language = "fr"
            elif "de" in language:
                language = "de"

            json_line = {
                Const.LANGUAGE: language,
                Const.FILE: input_file,
                Const.DATASET: dataset_name,
                Const.OCR: {
                    Const.LINE: Const.NONE,
                    Const.SENTENCE: ocr_sentence,
                    # Const.REGION: ocr_text,
                    "QA_Impresso_sentence": qa_bloom(ocr_sentence, language=language),
                    "QA_Pleias_sentence": qa_pleias(ocr_sentence, language=language),
                    "QA_Impresso_region": qa_bloom(ocr_text, language=language),
                    "QA_Pleias_region": qa_pleias(ocr_text, language=language),
                },  # TODO removed temporarily the region - too large
                Const.GROUND: {
                    Const.LINE: Const.NONE,
                    Const.SENTENCE: gs_sentence,
                    # Const.REGION: gt_text,
                    "QA_GT_Impresso_sentence": qa_bloom(gs_sentence, language=language),
                    "QA_GT_Pleias_sentence": qa_pleias(gs_sentence, language=language),
                    "QA_GT_Impresso_region": qa_bloom(gt_text, language=language),
                    "QA_GT_Pleias_region": qa_pleias(gt_text, language=language),
                },  # TODO removed temporarily the region - too large
            } | file_metadata

            # import pdb
            #
            # pdb.set_trace()
            # print(
            #     json_line[Const.OCR][Const.REGION][:20],
            #     "...",
            #     json_line[Const.OCR]["QA_Impresso_region"],
            #     json_line[Const.GROUND]["QA_GT_Impresso_region"],
            #     language,
            # )
            # from pprint import pprint
            #
            # pprint(json_line)
            outfile.write(json.dumps(json_line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text files and align sentences."
    )
    parser.add_argument(
        "--input_dir", help="The path to the input directory containing the text files."
    )
    parser.add_argument(
        "--output_dir",
        help="The path to the output directory where JSON Lines files will be created.",
    )
    parser.add_argument("--language", default="de", help="The language of the dataset.")
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()

    load_metadada(args)

    output_dir_path = args.input_dir.replace("original", "converted")

    dataset_name = args.input_dir.split("/")[-1]
    output_file = os.path.join(args.output_dir, "{}.jsonl".format(dataset_name))
    if os.path.exists(output_file):
        logging.info("{} already exists. It will be deleted.")
        os.remove(output_file)

    import glob

    files, langs = [], []
    logging.info("Writing output {}".format(output_file))
    for input_file in glob.glob(f"{args.input_dir}/**/*", recursive=True):
        if not os.path.isdir(input_file):
            # do something with the file
            if input_file.endswith(".txt") and "readme" not in input_file:

                logging.info("Analyzing file {}".format(input_file))
                if os.path.getsize(input_file) / 1024 <= 40:
                    # print(input_file, os.path.getsize(input_file) / 1024)
                    files.append(input_file)
                    langs.append(input_file.split("/")[-3])

    total_files = sum([len(files) for r, d, files in os.walk(args.input_dir)])
    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")

    print(f"There are {len(files)} files")
    for input_file in files:
        process_file(
            args=args,
            input_file=input_file,
            output_file=output_file,
            dataset_name=dataset_name,
        )
        progress_bar.update(1)
    progress_bar.close()

    from utils import print_statistics

    # Print statistics for both the full dataset and the train set
    # print_statistics(
    #     os.path.join(args.output_dir, "{}.jsonl".format(dataset_name)), dataset_name
    # )

    # TRAIN SET
    # output_file = os.path.join(args.output_dir, "{}-train.jsonl".format(dataset_name))
    # if os.path.exists(output_file):
    #     logging.info("{} already exists. It will be deleted.")
    #     os.remove(output_file)
    #
    # total_files = len(files)
    # progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")
    #
    # print(f"There are {len(files)} files")
    # for input_file in files:
    #     process_file(
    #         args=args,
    #         input_file=input_file,
    #         output_file=output_file,
    #         dataset_name=dataset_name,
    #     )
    #     progress_bar.update(1)
    # progress_bar.close()
