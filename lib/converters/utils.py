import pysbd
from genalog.text import anchor
from langdetect import detect
import re

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