import os
from bs4 import BeautifulSoup
import argparse
import textdistance
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
    segmenter = pysbd.Segmenter(language=language, clean=False)

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
            (clean_text(gt_sentence), clean_text(ocr_sentence)))

    return aligned_sentences


def process_file(
        input_file: str,
        ocr_file: str,
        output_file: str,
        extraction_type: str = 'region') -> None:

    # Parse the ground truth file
    with open(input_file, 'r') as f:
        gt_soup = BeautifulSoup(f, 'xml')

    aligned_texts = []
    # Parse the prediction file
    if not os.path.exists(ocr_file):
        print(f'{ocr_file} does not exist.')
    else:
        with open(ocr_file, 'r') as f:
            ocr_soup = BeautifulSoup(f, 'xml')

        # Iterate over each TextRegion in the ground truth file
        for gt_region in gt_soup.find_all('TextRegion'):
            # Find the corresponding TextRegion in the prediction file
            region_id = gt_region['id']
            ocr_region = ocr_soup.find('TextRegion', {'id': region_id})

            # Extract the TextEquiv content for the TextRegion in the ground
            # truth and prediction files
            gt_region_text = gt_region.findAll('TextEquiv')[-1].text.strip()

            ocr_region_text = None
            try:
                ocr_region_text = ocr_region.findAll(
                    'TextEquiv')[-1].text.strip()
            except BaseException:
                print(f'{region_id} not found in {ocr_file}')
            # Print the extracted TextEquiv content for the TextRegion in the
            # ground truth and prediction files

            # If the extraction type is "line", iterate over each TextLine in
            # the TextRegion
            if extraction_type == 'line':
                # aligned_lines = []
                for gt_line in gt_region.find_all('TextLine'):
                    # Find the corresponding TextLine in the prediction file
                    line_id = gt_line['id']
                    ocr_line = ocr_soup.find('TextLine', {'id': line_id})

                    # Extract the TextEquiv content for the TextLine in the
                    # ground truth and prediction files
                    gt_line_text = gt_line.find('TextEquiv').text
                    pred_line_text = ocr_line.find('TextEquiv').text

                    # Print the extracted TextEquiv content for the TextLine in the ground truth and
                    # prediction files
                    print(
                        f'Ground truth TextLine {line_id}:\n{gt_line_text}\n')
                    print(
                        f'Prediction TextLine {line_id}:\n{pred_line_text}\n')

                    aligned_texts.append((gt_line, ocr_line))

            # If the extraction type is "region", skip the TextLine iteration
            elif extraction_type == 'region':

                if ocr_region_text:
                    # Add the already aligned regions
                    aligned_texts += [(gt_region_text, ocr_region_text)]

                # If the extraction type is "region", skip the TextLine
                # iteration
            elif extraction_type == 'sentence':

                if ocr_region_text:
                    # Align the OCR and GS sentences
                    aligned_texts += align_texts(gt_region_text,
                                                 ocr_region_text, language=args.language)

            # Raise an error if the extraction type is invalid
            else:
                raise ValueError(
                    f'Invalid extraction type "{extraction_type}"')

        # Write the output to a JSON Lines file
        with open(output_file, "w") as outfile:
            for ocr_text, gs_text in aligned_texts:
                json_line = json.dumps(
                    {"ocr_text": ocr_text, "correct_text": gs_text})
                outfile.write(json_line + "\n")


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(
        description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to ground truth folder')
    parser.add_argument('--ocr_dir', type=str, help='Path to OCRed folder')
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

    output_dir_path = args.input_dir.replace('original', 'converted')

    output_file = os.path.join(args.output_dir, '{}.jsonl'.format(args.input_dir.split('/')[-1]))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".xml"):
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(args=args, input_file=input_file, output_file=output_file)
                progress_bar.update(1)
    progress_bar.close()