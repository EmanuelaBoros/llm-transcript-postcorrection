import difflib
from nltk.tokenize import sent_tokenize
import Levenshtein
import numpy as np
from fastdtw import fastdtw
from nltk.tokenize import word_tokenize


def process_text(text):
    cleaned_text = text.strip()
    return cleaned_text


def character_level_alignment(text1, text2):
    text1 = process_text(text1)
    text2 = process_text(text2)

    # Perform the character-level alignment
    matcher = difflib.SequenceMatcher(None, text1, text2)
    aligned_text = "".join([text1[i:i + n]
                           for i, j, n in matcher.get_matching_blocks() if n > 0])

    # Tokenize the aligned text into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(aligned_text)

    return list(zip(sentences1, sentences2))


def distance_matrix(words1, words2):
    n = len(words1)
    m = len(words2)
    dist_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = Levenshtein.distance(words1[i], words2[j])

    return dist_matrix


def word_level_alignment(text1, text2):
    text1 = process_text(text1)
    text2 = process_text(text2)

    words1 = word_tokenize(text1)
    words2 = word_tokenize(text2)

    dist_matrix = distance_matrix(words1, words2)
    _, path = fastdtw(dist_matrix)

    aligned_text1 = []
    aligned_text2 = []

    for i, j in path:
        aligned_text1.append(words1[i])
        aligned_text2.append(words2[j])

    return " ".join(aligned_text1), " ".join(aligned_text2)


text1 = "O R D Tyrawley, who lately ar r ived here, is exceedingly ... well received." \
        " He contejgtfaily with the MinnSrs, and, 'tis thought, will fpaediiy carry h.s Point."
text2 = "LORD Tyrawley, who lately arrived here, is exceedingly well received. He confers daily with the  and, 'tis " \
        "thought, will impeedily carry his Point."

aligned_sentences = character_level_alignment(text1, text2)

for sentence1, sentence2 in aligned_sentences:
    print(f"Text1: {sentence1}\nText2: {sentence2}\n")

print(word_level_alignment(text1, text2))
