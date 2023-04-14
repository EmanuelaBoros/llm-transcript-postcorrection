import os
import json
import re
import argparse
from fastdtw import fastdtw
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from scipy.spatial.distance import euclidean


def process_text(text):
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()

    # Split the text into sentences
    sentences = sent_tokenize(cleaned_text)

    return sentences


def align_sentences(sentences_ocr, sentences_gs):
    """
    :param sentences_ocr: 
    :param sentences_gs:
    :return:
    """
    aligned_sentences = []

    # Tokenize the sentences into words
    tokenized_ocr = [word_tokenize(s) for s in sentences_ocr]
    tokenized_gs = [word_tokenize(s) for s in sentences_gs]

    # Perform the alignment using DTW
    _, path = fastdtw(tokenized_ocr, tokenized_gs, dist=euclidean)

    for idx_ocr, idx_gs in path:
        aligned_sentences.append((sentences_ocr[idx_ocr], sentences_gs[idx_gs]))

    return aligned_sentences


def process_file(input_file, output_file):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    # Extract OCR and GS sentences from the data list
    ocr_sentences = process_text(data[0])
    gs_sentences = process_text(data[2])

    # Align the OCR and GS sentences
    aligned_sentences = align_sentences(ocr_sentences, gs_sentences)

    # Write the output to a JSON Lines file
    with open(output_file, "w") as outfile:
        for ocr_sentence, gs_sentence in aligned_sentences:
            json_line = json.dumps({"ocr_text": ocr_sentence, "correct_text": gs_sentence})
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files and align sentences.")
    parser.add_argument("--input_dir", help="The path to the input directory containing the text files.")
    parser.add_argument("--output_dir", help="The path to the output directory where JSON Lines files will be created.")

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            # Check if the file has the desired format (e.g., .txt)
            if file.endswith(".txt"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(args.output_dir, file.replace(".txt", ".jsonl"))

                process_file(input_file, output_file)