import os
import argparse
import logging
from tqdm import tqdm
import json
import pysbd
from genalog.text import anchor


def clean_text(text):
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace('@', '')
    return cleaned_text


def align_texts(gt_text, ocr_text, language='en'):
    segmenter = pysbd.Segmenter(language=language, article_id=None)

    # We align the texts with RETAS Method
    aligned_gt, aligned_noise = anchor.align_w_anchor(gt_text, ocr_text)
    # print(f"Original: {gt_text}")
    # print(f"OCR: {ocr_text}")
    # print(f"Aligned ground truth: {aligned_gt}")
    # print(f"Aligned noise:        {aligned_noise}\n")

    # We split the ground truth sentences and we consider them as the
    # "correct" tokenization
    gt_sentences = segmenter.segment(aligned_gt)

    # We split the noisy text following the sentences' lengths in the ground
    # truth
    sentence_lengths = [len(sentence) for sentence in gt_sentences]

    ocr_sentences = []
    start = 0

    for length in sentence_lengths:
        end = start + length
        ocr_sentences.append(aligned_noise[start:end])
        start = end

    assert len(gt_sentences) == len(ocr_sentences)

    aligned_sentences = []
    # Clean the sentences from the alignment characters @
    for gt_sentence, ocr_sentence in zip(gt_sentences, ocr_sentences):
        aligned_sentences.append(
            (clean_text(gt_sentence), clean_text(ocr_sentence)), article_id)

    return aligned_sentences


def process_file(
        input_file: str,
        output_file: str,
        extraction_type: str = 'region') -> None:

    # Parse the ground truth file
    with open(input_file, 'r') as f:
        text = f.read()

    articles = text.split('*$*OVERPROOF*$*')

    aligned_texts = []

    for article in articles:
        if not article.strip():
            continue
        lines = article.split('\n')

        # Keep the article id
        article_id = lines[0].strip()

        aligned_article_lines = []
        # Align the lines before all types of extraction so the region/article
        # can be produced
        for line in lines:
            if '||@@||' in line:
                aligned_lines = line.split('||@@||') + [article_id]
                aligned_article_lines.append(tuple(aligned_lines))

        if extraction_type == 'line':
            aligned_texts += aligned_article_lines

        # The region in OVERPROOF is the article level
        elif extraction_type == 'region':
            gt_region_text = ' '.join(
                [gt_line for gt_line, _ in aligned_article_lines]).strip()
            ocr_region_text = ' '.join(
                [ocr_line for _, ocr_line in aligned_article_lines]).strip()
            aligned_texts.append((gt_region_text, ocr_region_text, article_id))

        elif extraction_type == 'sentence':
            gt_region_text = ' '.join(
                [gt_line for gt_line, _ in aligned_article_lines]).strip()
            ocr_region_text = ' '.join(
                [ocr_line for _, ocr_line in aligned_article_lines]).strip()
            aligned_texts += align_texts(gt_region_text,
                                         ocr_region_text,
                                         language=args.language,
                                         article_id=article_id)

        else:
            raise ValueError(
                "Invalid extraction_type. Choose between 'line', 'region', or 'sentence'.")

    # Write the output to a JSON Lines file
    with open(output_file, "w") as outfile:
        for text in aligned_texts:
            ocr_text, gs_text, article_id = text[0], text[1], text[-1]
            json_line = json.dumps(
                {"ocr_text": ocr_text, "correct_text": gs_text, 'article_id': article_id})
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(
        description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to ground truth folder')
    parser.add_argument(
        '--extraction_type',
        type=str,
        default='region',
        choices=[
            'region',
            'line',
            'sentence'],
        help='Specify whether to extract the TextEquiv content per region or per line')
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

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))
                # Compute the output file path by replacing the input directory
                # with the output directory
                output_file = input_file.replace(
                    'original',
                    'converted').replace(
                    file,
                    os.path.join(
                        args.extraction_type,
                        file)).replace(
                    ".txt",
                    ".jsonl")
                logging.info('Writing output {}'.format(output_file))
                # Create the output directory if it does not exist
                output_dir_path = os.path.dirname(output_file)

                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                process_file(input_file, output_file,
                             extraction_type=args.extraction_type)
                progress_bar.update(1)
    progress_bar.close()
