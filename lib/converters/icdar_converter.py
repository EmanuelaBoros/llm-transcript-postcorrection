import os
import json
import re
import argparse
from fastdtw import fastdtw
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.spatial.distance import cdist
import textdistance


def process_text(text):
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()

    # Split the text into sentences
    sentences = sent_tokenize(cleaned_text)

    return sentences

def custom_similarity(sentence1, sentence2):
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


def process_file(input_file, output_file):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    # Extract OCR and GS sentences from the data list
    # [OCR_toInput] [OCR_aligned] [ GS_aligned]
    # import pdb;pdb.set_trace()
    ocr_sentences = process_text(data[0].replace('[OCR_toInput]', '').strip())
    gs_sentences = process_text(data[2].replace('[ GS_aligned]', '').strip())

    # Align the OCR and GS sentences
    aligned_sentences = align_sentences(ocr_sentences, gs_sentences)

    # Write the output to a JSON Lines file
    # import pdb;pdb.set_trace()
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
        for file in tqdm(files, total=len(files)):
            if file.endswith(".txt") and 'readme' not in file:
                input_file = os.path.join(root, file)
                print('Analyzing file {}'.format(input_file))
                # Compute the output file path by replacing the input directory with the output directory
                output_file = os.path.join(args.output_dir,
                                           os.path.relpath(input_file, args.input_dir)).replace(".txt", ".jsonl")
                print('Writing output {}'.format(output_file))
                # Create the output directory if it does not exist
                output_dir_path = os.path.dirname(output_file)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                process_file(input_file, output_file)