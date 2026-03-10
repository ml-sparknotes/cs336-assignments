from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def word_count_from_document(document: str) -> Counter:
    words = Counter()
    doc_words = re.finditer(PAT, document)
    for word in doc_words:
        token_seq = []
        for byte in word.group().encode("utf-8"):
            token_seq.append(byte)
        words[tuple(token_seq)] += 1
    return words


def word_count_from_document_batch(documents: list[str]) -> Counter:
    words = Counter()
    for document in documents:
        words.update(word_count_from_document(document))
    return words



