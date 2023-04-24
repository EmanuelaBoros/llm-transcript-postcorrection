import pysbd
from genalog.text import anchor
import re
import warnings
warnings.filterwarnings('ignore')


def clean_text(text):
    """
    :param text:
    :return:
    """
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()
    cleaned_text = re.sub(r"@+", "", cleaned_text).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('¬ ', '')
    cleaned_text = cleaned_text.replace('¬\n', '')

    # If ¬ is still in the sentence:
    cleaned_text = cleaned_text.replace('¬', '')
    return cleaned_text


def align_texts(gt_text, ocr_text, language='en'):
    gt_text = clean_text(gt_text)
    ocr_text = clean_text(ocr_text)

    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except BaseException:
        # Defaulting to en if a tokenizer is not available in a specific
        # language
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


# Function to reconstruct sentences from text lines
def reconstruct_text(txt_lines, sentences):
    reconstructed_text = " ".join(txt_lines)
    line_index_mapping = []
    sentence_index_mapping = []

    for line in txt_lines:
        start_index = reconstructed_text.find(line)

        if start_index != -1:
            end_index = start_index + len(line)
            line_index_mapping.append({"line": line, "start_index": start_index, "end_index": end_index})

    for sentence in sentences:
        start_index = reconstructed_text.find(sentence)

        if start_index != -1:
            end_index = start_index + len(sentence)
            sentence_index_mapping.append({"sentence": sentence, "start_index": start_index, "end_index": end_index})

    return reconstructed_text, line_index_mapping, sentence_index_mapping


def map_line_to_sentence(line_index_mapping, sentence_index_mapping):
    mapping = []

    for line_index in line_index_mapping:
        line_mapping = None
        max_overlap = -1

        for sentence_index in sentence_index_mapping:
            overlap_start = max(line_index["start_index"], sentence_index["start_index"])
            overlap_end = min(line_index["end_index"], sentence_index["end_index"])
            overlap = overlap_end - overlap_start + 1
            line_coverage = overlap / (line_index["end_index"] - line_index["start_index"] + 1)

            if line_coverage > 0.5 and overlap > max_overlap:
                max_overlap = overlap
                line_mapping = (line_index["line"], sentence_index["sentence"])

        if line_mapping is None:
            closest_sentence = None
            min_distance = float("inf")
            for sentence_index in sentence_index_mapping:
                distance = min(abs(line_index["start_index"] - sentence_index["start_index"]),
                               abs(line_index["end_index"] - sentence_index["end_index"]))
                if distance < min_distance:
                    min_distance = distance
                    closest_sentence = sentence_index["sentence"]
            line_mapping = (line_index["line"], closest_sentence)

        if line_mapping[1] is None:
            for sentence_index in sentence_index_mapping:
                if line_index["start_index"] >= sentence_index["start_index"] and line_index["end_index"] <= sentence_index["end_index"]:
                    line_mapping = (line_index["line"], sentence_index["sentence"])
                    break

        mapping.append(line_mapping)

    return mapping
