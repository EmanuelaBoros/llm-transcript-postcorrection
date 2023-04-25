import os
import json
import argparse
from tqdm import tqdm
import logging
from langdetect import detect
from utils import clean_text, align_texts
from const import Const
import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_metadada(args):
    metadata_path = None
    for path in glob.glob(args.input_dir + '/**/*', recursive=True):
        if 'eval_metadata' in path:
            metadata_path = path

    if metadata_path is None:
        print('Metadata was not found.')
        args.metadata = None
    else:
        with open(metadata_path, 'r') as f:
            metadata = f.readlines()

        metadata = [line.strip().split(';') for line in metadata]
        # Skip columns line: 'File;Date;Type;NbAlignedChar'
        columns, metadata = metadata[0], metadata[1:]

        metadata = pd.DataFrame(metadata, columns=columns)
        metadata['File'] = metadata['File'].apply(
            lambda x: x.replace(
                '\\', '/'))  # Replace Windows style file names

        args.metadata = metadata


def lookup_metadata(args, input_file):
    file_metadata = args.metadata[args.metadata.File ==
                                  '/'.join(input_file.split('/')[-2:])]
    return file_metadata.to_dict('records')[0]


def process_file(args, input_file, output_file):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    print(input_file)

    if args.metadata is not None:
        file_metadata = lookup_metadata(args, input_file)
    else:
        file_metadata = {}

    # Extract OCR and GS sentences from the data list
    # [OCR_toInput] [OCR_aligned] [ GS_aligned]
    gt_text = clean_text(data[2].replace('[ GS_aligned]', '').strip())
    ocr_text = clean_text(data[0].replace('[OCR_toInput]', '').strip())

    try:
        language = detect(gt_text)
    except:
        language = 'en'

    try:
        # Align the OCR and GS sentences
        aligned_sentences = align_texts(gt_text, ocr_text, language=language)
    except BaseException:
        # Defaulting to English
        aligned_sentences = align_texts(gt_text, ocr_text, language='en')

    # Write the output to a JSON Lines file
    with open(output_file, "a") as outfile:
        for ocr_sentence, gs_sentence in aligned_sentences:
            json_line = json.dumps({Const.FILE: input_file,
                                    Const.OCR: {Const.LINE: Const.NONE,
                                                Const.SENTENCE: ocr_sentence,
                                                Const.REGION: ocr_text},  # TODO removed temporarily the region - too large
                                    Const.GROUND: {Const.LINE: Const.NONE,
                                                   Const.SENTENCE: gs_sentence,
                                                   Const.REGION: gt_text}  # TODO removed temporarily the region - too large
                                    } | file_metadata)

            outfile.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text files and align sentences.")
    parser.add_argument(
        "--input_dir",
        help="The path to the input directory containing the text files.")
    parser.add_argument(
        "--output_dir",
        help="The path to the output directory where JSON Lines files will be created.")
    parser.add_argument(
        "--language", default='de',
        help="The language of the dataset.")
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

    total_files = sum([len(files) for r, d, files in os.walk(args.input_dir)])
    progress_bar = tqdm(
        total=total_files,
        desc="Processing files",
        unit="file")

    load_metadada(args)

    output_dir_path = args.input_dir.replace('original', 'converted')

    output_file = os.path.join(args.output_dir,
                               '{}.jsonl'.format(args.input_dir.split('/')[-1]))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt") and 'readme' not in file:
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(
                    args=args,
                    input_file=input_file,
                    output_file=output_file)
                progress_bar.update(1)
    progress_bar.close()
