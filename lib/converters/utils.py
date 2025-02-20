import pysbd
from genalog.text import anchor
import re
import warnings

warnings.filterwarnings("ignore")


def clean_text(text):
    """
    :param text:
    :return:
    """
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()
    cleaned_text = re.sub(r"@+", "", cleaned_text).strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.replace("¬ ", "")
    cleaned_text = cleaned_text.replace("¬\n", "")

    # If ¬ is still in the sentence:
    cleaned_text = cleaned_text.replace("¬", "")
    return cleaned_text


def align_texts(gt_text, ocr_text, language="en"):
    gt_text = clean_text(gt_text)
    ocr_text = clean_text(ocr_text)

    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except BaseException:
        # Defaulting to en if a tokenizer is not available in a specific
        # language
        segmenter = pysbd.Segmenter(language="en", clean=False)

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
        aligned_sentences.append((clean_text(gt_sentence), clean_text(ocr_sentence)))

    return aligned_sentences


# Function to reconstruct sentences from text lines
def reconstruct_text(txt_lines, sentences):
    reconstructed_text = " ".join(txt_lines)
    line_index_mapping = []
    sentence_index_mapping = []

    for line in txt_lines:
        start_index = reconstructed_text.find(line)

        if start_index != -1:
            end_index = start_index + len(line)
            line_index_mapping.append(
                {"line": line, "start_index": start_index, "end_index": end_index}
            )

    for sentence in sentences:
        start_index = reconstructed_text.find(sentence)

        if start_index != -1:
            end_index = start_index + len(sentence)
            sentence_index_mapping.append(
                {
                    "sentence": sentence,
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

    return reconstructed_text, line_index_mapping, sentence_index_mapping


def map_lines_to_sentences(lines, sentences, ocr_lines, ocr_sentences):
    line_index_mapping = {}
    sentence_index_mapping = {}
    result = []
    ocr_result = []
    for i, line in enumerate(lines):
        if i not in line_index_mapping:
            for j, sentence in enumerate(sentences):
                if line in sentence:
                    sentence_index_mapping[j] = sentence
                    result.append((line, sentence))
                    ocr_result.append((ocr_lines[i], ocr_sentences[j]))
                    break
                elif sentence in line:
                    sentence_index_mapping[j] = sentence
                    result.append((line, sentence))
                    ocr_result.append((ocr_lines[i], ocr_sentences[j]))
                    break

    for i, sentence in enumerate(sentences):
        sentence_index_mapping[i] = sentence
        start = 0
        for j, line in enumerate(lines):
            if sentence in line:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))
                start = len(line)
            elif start > 0 and line in sentence[start:]:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))
                start = 0
            elif sentence in line:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))

    return result, ocr_result


from collections import defaultdict
from const import Const
import json
import os
import csv
from collections import defaultdict
from colorama import Fore, Style


def update_csv(stats, csv_file):
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)


def print_statistics(output_file, dataset_name, csv_file="dataset_statistics.csv"):
    total_files = 0
    total_sentences = 0
    total_lines = 0
    total_regions = set()
    total_words_ocr = 0
    total_words_ground = 0
    total_chars_ocr = 0
    total_chars_ground = 0
    language_counts = defaultdict(int)

    with open(output_file, "r") as infile:
        for line in infile:
            total_files += 1
            data = json.loads(line)
            language_counts[data[Const.LANGUAGE]] += 1
            total_sentences += 1  # Each JSON entry corresponds to a sentence
            if data[Const.OCR][Const.LINE] != Const.NONE:
                total_lines += 1
            total_words_ocr += len(data[Const.OCR][Const.SENTENCE].split())
            total_words_ground += len(data[Const.GROUND][Const.SENTENCE].split())
            total_chars_ocr += len(data[Const.OCR][Const.SENTENCE])
            total_chars_ground += len(data[Const.GROUND][Const.SENTENCE])
            total_regions.add(data[Const.OCR][Const.REGION])  # Avoid duplication

    stats = {
        "Dataset": dataset_name,
        "Total Files Processed": total_files,
        "Total Sentences": total_sentences,
        "Total Lines": total_lines,
        "Unique Regions": len(total_regions),
        "Total Words in OCR": total_words_ocr,
        "Total Words in Ground Truth": total_words_ground,
        "Total Characters in OCR": total_chars_ocr,
        "Total Characters in Ground Truth": total_chars_ground,
    }

    # Add language distribution to stats
    for lang, count in language_counts.items():
        stats[f"Lang-{lang}"] = count

    # Update the CSV file with the statistics
    update_csv(stats, csv_file)

    print(
        f"\n{Fore.CYAN}=== 📊 Dataset Statistics for {dataset_name.upper()} ==={Style.RESET_ALL}"
    )
    print(f"{Fore.GREEN}📁 Total Files Processed: {total_files:,}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📝 Total Sentences: {total_sentences:,}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📄 Total Lines: {total_lines:,}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}🌍 Unique Regions: {len(total_regions):,}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔠 Total Words in OCR: {total_words_ocr:,}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}📖 Total Words in Ground Truth: {total_words_ground:,}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.MAGENTA}🔤 Total Characters in OCR: {total_chars_ocr:,}{Style.RESET_ALL}"
    )
    print(
        f"{Fore.MAGENTA}🔡 Total Characters in Ground Truth: {total_chars_ground:,}{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}🌎 Language Distribution:{Style.RESET_ALL}")
    for lang, count in language_counts.items():
        print(f"  {Fore.LIGHTYELLOW_EX}📌 {lang}: {count}{Style.RESET_ALL}")
