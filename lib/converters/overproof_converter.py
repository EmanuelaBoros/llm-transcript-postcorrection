import os
import argparse
import logging
from tqdm import tqdm
import json
from langdetect import detect
import re
from const import Const
from utils import clean_text, align_texts, reconstruct_sentences


def process_file(args,
                 input_file: str,
                 output_file: str,
                 extraction_type: str = 'region') -> None:

    # Parse the ground truth file
    with open(input_file, 'r') as f:
        text = f.read()

    articles = text.split('*$*OVERPROOF*$*')

    aligned_texts = []

    language = detect(text)

    for article in articles:
        if not article.strip():
            continue
        lines = article.split('\n')

        # Keep the article id
        article_id = lines[0].strip()

        # Align the lines before all types of extraction so the region/article
        # can be produced
        aligned_lines = []
        for line in lines:
            line = line.strip()
            if '||@@||' in line:
                aligned_lines.append(tuple(line.split('||@@||')))

        # import pdb;pdb.set_trace()
        # The region in OVERPROOF is the article level
        gt_region_text = ' '.join(
            [gt_line for gt_line, _ in aligned_lines]).strip()
        ocr_region_text = ' '.join(
            [ocr_line for _, ocr_line in aligned_lines]).strip()
        aligned_texts.append((gt_region_text, ocr_region_text, article_id))

        # Split in sentences and align
        aligned_sentences = align_texts(gt_region_text,
                                        ocr_region_text,
                                        language=language)

        gt_reconstructed_sentences = reconstruct_sentences([gt_line for gt_line, _ in aligned_lines],
                                                           [gt_sentence for gt_sentence, _ in aligned_sentences])
        ocr_reconstructed_sentences = reconstruct_sentences([ocr_line for _, ocr_line in aligned_lines],
                                                            [ocr_sentence for _, ocr_sentence in aligned_sentences])

        try:
            assert len(gt_reconstructed_sentences) == len(ocr_reconstructed_sentences)
        except BaseException:
            import pdb
            pdb.set_trace()

        # Append the output to a JSON Lines file
        with open(output_file, "a") as outfile:
            for gt_reconstructed_sentence, gt_line, ocr_reconstructed_sentence, ocr_line in zip(
                gt_reconstructed_sentences, [gt_line for gt_line, _ in aligned_lines], ocr_reconstructed_sentences,
                    [ocr_line for _, ocr_line in aligned_lines]):

                json_line = json.dumps({Const.FILE: input_file,
                                        Const.OCR: {Const.LINE: clean_text(ocr_line),
                                                    Const.SENTENCE: clean_text(ocr_reconstructed_sentence),
                                                    Const.REGION: clean_text(ocr_region_text)},
                                        # TODO removed temporarily the region - too
                                        # large
                                        Const.GROUND: {Const.LINE: clean_text(gt_line),
                                                       Const.SENTENCE: clean_text(gt_reconstructed_sentence),
                                                       Const.REGION: clean_text(gt_region_text)}
                                        # TODO removed temporarily the region - too
                                        # large
                                        } | {'article_id': article_id})
                outfile.write(json_line + "\n")
                outfile.flush()


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(
        description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to ground truth folder')
    parser.add_argument(
        "--output_dir",
        help="The path to the output directory where JSON Lines files will be created.")
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

    output_dir_path = args.input_dir.replace('original', 'converted')

    output_file = os.path.join(args.output_dir,
                               '{}.jsonl'.format(args.input_dir.split('/')[-1]))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(
                    args=args,
                    input_file=input_file,
                    output_file=output_file)
                progress_bar.update(1)
    progress_bar.close()
