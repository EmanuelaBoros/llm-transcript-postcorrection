import os
from bs4 import BeautifulSoup
import argparse
from typing import Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
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
                print(similarity)
                best_match = (sentence2, similarity)
        aligned_sentences.append((sentence1, best_match[0]))

    return aligned_sentences


def process_text(text):
    cleaned_text = text.strip()
    words = word_tokenize(cleaned_text)
    return words


def custom_similarity(word1, word2):
    return textdistance.jaro_winkler(word1, word2)


def align_texts(text1, text2):
    words1 = process_text(text1)
    words2 = process_text(text2)

    aligned_words = textdistance.dtw(words1, words2, lambda w1, w2: 1 - custom_similarity(w1, w2))

    aligned_sentences = []
    sentence1 = []
    sentence2 = []

    for word1, word2 in aligned_words.path:
        if word1 is not None:
            sentence1.append(words1[word1])
        if word2 is not None:
            sentence2.append(words2[word2])

        if word1 is not None and word2 is not None and (words1[word1] == '.' or words2[word2] == '.'):
            aligned_sentences.append((' '.join(sentence1).strip(), ' '.join(sentence2).strip()))
            sentence1 = []
            sentence2 = []

    if sentence1 or sentence2:
        aligned_sentences.append((' '.join(sentence1).strip(), ' '.join(sentence2).strip()))

    return aligned_sentences


def process_file(input_file, ocr_file, output_file, extraction_type):

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

                # Extract the TextEquiv content for the TextRegion in the ground truth and prediction files
                gt_region_text = gt_region.findAll('TextEquiv')[-1].text.strip()

                ocr_region_text = None
                try:
                    ocr_region_text = ocr_region.findAll('TextEquiv')[-1].text.strip()
                except:
                    print(f'{region_id} not found in {ocr_file}')
                # Print the extracted TextEquiv content for the TextRegion in the ground truth and prediction files

                # If the extraction type is "line", iterate over each TextLine in the TextRegion
                if extraction_type == 'line':
                    # aligned_lines = []
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

                        aligned_texts.append((gt_line, ocr_line))

                # If the extraction type is "region", skip the TextLine iteration
                elif extraction_type == 'region':

                    if ocr_region_text:
                        # Align the OCR and GS sentences
                        aligned_texts += [(gt_region_text, ocr_region_text)]

                    # If the extraction type is "region", skip the TextLine iteration
                elif extraction_type == 'sentence':

                    gt_region_sentences = sent_tokenize(gt_region_text)
                    if ocr_region_text:
                        pred_region_sentences = sent_tokenize(ocr_region_text)

                        # Align the OCR and GS sentences
                        aligned_texts += align_texts(gt_region_text, ocr_region_text)

                # Raise an error if the extraction type is invalid
                else:
                    raise ValueError(f'Invalid extraction type "{extraction_type}"')

            # Write the output to a JSON Lines file
            with open(output_file, "w") as outfile:
                for ocr_text, gs_text in aligned_texts:
                    json_line = json.dumps({"ocr_text": ocr_text, "correct_text": gs_text})
                    outfile.write(json_line + "\n")


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument('--input_dir', type=str, help='Path to ground truth folder')
    parser.add_argument('--ocr_dir', type=str, help='Path to OCRed folder')
    parser.add_argument('--extraction_type', type=str, default='region', choices=['region', 'line', 'sentence'],
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