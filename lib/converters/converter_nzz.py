import os
from bs4 import BeautifulSoup
import argparse
from typing import Tuple
from nltk.tokenize import sent_tokenize
import textdistance
import logging
from tqdm import tqdm
import json


def custom_similarity(sentence1, sentence2):
    """
    :param sentence1:
    :param sentence2:
    :return:
    """
    return textdistance.jaccard(sentence1, sentence2)


def align_sentences(sentences1, sentences2):

    aligned_sentences = []
    for sentence1 in sentences1:
        best_match = (None, 0)
        for sentence2 in sentences2:



            similarity = custom_similarity(sentence1, sentence2)
            if similarity > best_match[1]:
                best_match = (sentence2, similarity)
        aligned_sentences.append((sentence1, best_match[0]))

    return aligned_sentences


def process_file(input_file, ocr_file, output_file, extraction_type):

        # Parse the ground truth file
        with open(input_file, 'r') as f:
            gt_soup = BeautifulSoup(f, 'xml')

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

                # Extract the TextEquiv content for the TextRegion in the ground truth and prediction files
                gt_region_text = gt_region.find('TextEquiv').text.strip()

                ocr_region_text = None
                try:
                    ocr_region_text = ocr_region.find('TextEquiv').text.strip()
                except:
                    print(f'{region_id} not found in {ocr_file}')
                # Print the extracted TextEquiv content for the TextRegion in the ground truth and prediction files

                # If the extraction type is "line", iterate over each TextLine in the TextRegion
                if extraction_type == 'line':
                    aligned_lines = []
                    for gt_line in gt_region.find_all('TextLine'):
                        # Find the corresponding TextLine in the prediction file
                        line_id = gt_line['id']
                        ocr_line = ocr_soup.find('TextLine', {'id': line_id})

                        # Extract the TextEquiv content for the TextLine in the ground truth and prediction files
                        gt_line_text = gt_line.find('TextEquiv').text
                        pred_line_text = ocr_line.find('TextEquiv').text

                        # Print the extracted TextEquiv content for the TextLine in the ground truth and
                        # prediction files
                        print(f'Ground truth TextLine {line_id}:\n{gt_line_text}\n')
                        print(f'Prediction TextLine {line_id}:\n{pred_line_text}\n')

                        aligned_lines.append((gt_line, ocr_line))
                    # Write the output to a JSON Lines file
                    with open(output_file, "w") as outfile:
                        for ocr_line, gs_line in aligned_lines:
                            json_line = json.dumps({"ocr_text": ocr_line, "correct_text": gs_line})
                            outfile.write(json_line + "\n")

                # If the extraction type is "region", skip the TextLine iteration
                elif extraction_type == 'region':

                    gt_region_sentences = sent_tokenize(gt_region_text)
                    if ocr_region_text:
                        pred_region_sentences = sent_tokenize(ocr_region_text)

                        # Align the OCR and GS sentences
                        aligned_sentences = align_sentences(gt_region_sentences, pred_region_sentences)

                        # Write the output to a JSON Lines file
                        with open(output_file, "w") as outfile:
                            for ocr_sentence, gs_sentence in aligned_sentences:
                                json_line = json.dumps({"ocr_text": ocr_sentence, "correct_text": gs_sentence})
                                outfile.write(json_line + "\n")

                # Raise an error if the extraction type is invalid
                else:
                    raise ValueError(f'Invalid extraction type "{extraction_type}"')


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument('--input_dir', type=str, help='Path to ground truth folder')
    parser.add_argument('--ocr_dir', type=str, help='Path to OCRed folder')
    parser.add_argument('--extraction_type', type=str, default='region', choices=['region', 'line'],
                        help='Specify whether to extract the TextEquiv content per region or per line')
    parser.add_argument("--output_dir", help="The path to the output directory where JSON Lines files will be created.")
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
    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file")

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".xml"):
                input_file = os.path.join(root, file)
                ocr_file = os.path.join(args.ocr_dir, file)

                logging.info('Analyzing file {}'.format(input_file))
                # Compute the output file path by replacing the input directory with the output directory
                output_file = os.path.join(args.output_dir,
                                           os.path.relpath(input_file, args.input_dir)).replace(".txt", ".jsonl")
                logging.info('Writing output {}'.format(output_file))
                # Create the output directory if it does not exist
                output_dir_path = os.path.dirname(output_file)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                process_file(input_file, ocr_file, output_file, extraction_type=args.extraction_type)
                progress_bar.update(1)
    progress_bar.close()