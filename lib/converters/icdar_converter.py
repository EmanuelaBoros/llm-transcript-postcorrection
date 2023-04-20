import os
import json
import re
import argparse
from tqdm import tqdm
import logging
import pysbd
from genalog.text import anchor
from langdetect import detect

def process_text(text):
    """
    :param text:
    :return:
    """
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()
    cleaned_text = re.sub(r"@+", "", cleaned_text).strip()

    return cleaned_text

def clean_text(text):
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace('@', '')
    return cleaned_text


def align_texts(gt_text, ocr_text, language='en'):

    gt_text = process_text(gt_text)
    ocr_text = process_text(ocr_text)

    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except:
        # Defaulting to en if a tokenizer is not available in a specific language
        segmenter = pysbd.Segmenter(language='en', clean=False)
        
    # We align the texts with RETAS Method
    aligned_gt, aligned_noise = anchor.align_w_anchor(gt_text, ocr_text)

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


def process_file(input_file, output_file):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    # Extract OCR and GS sentences from the data list
    # [OCR_toInput] [OCR_aligned] [ GS_aligned]
    gt_text = process_text(data[2].replace('[ GS_aligned]', '').strip())
    ocr_text = process_text(data[0].replace('[OCR_toInput]', '').strip())

    language = detect(gt_text)
    # Align the OCR and GS sentences
    aligned_sentences = align_texts(gt_text, ocr_text, language=language)

    # Write the output to a JSON Lines file
    with open(output_file, "w") as outfile:
        for ocr_sentence, gs_sentence in aligned_sentences:
            json_line = json.dumps({"ocr_text": ocr_sentence, "correct_text": gs_sentence})
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and align sentences.")
    parser.add_argument("--input_dir", help="The path to the input directory containing the text files.")
    parser.add_argument("--output_dir", help="The path to the output directory where JSON Lines files will be created.")
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
            if file.endswith(".txt") and 'readme' not in file:
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