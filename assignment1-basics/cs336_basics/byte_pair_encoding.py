from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import regex as re
from tqdm import tqdm
import time
from collections import Counter

try:
    from cs336_basics.bpe_workers import word_count_from_document, word_count_from_document_batch
except ModuleNotFoundError:
    from bpe_workers import word_count_from_document, word_count_from_document_batch


def contains_subtuple(t, sub):
    n, m = len(t), len(sub)
    return any(t[i:i+m] == sub for i in range(n - m + 1))


def replace_pair(t: tuple[int, ...], pair: tuple[int, int], new_val: int) -> tuple[int, ...]:
    a, b = pair
    result = []
    i = 0
    n = len(t)

    while i < n:
        # Check if pair matches at current position
        if i < n - 1 and t[i] == a and t[i + 1] == b:
            result.append(new_val)
            i += 2  # Skip both elements of the pair
        else:
            result.append(t[i])
            i += 1

    return tuple(result)


def populate_words(input_path: str, special_tokens: list[str], reverse_vocab: dict[bytes, int], num_workers: int | None = None) -> dict[tuple[int], int]:
    # 1. Split documents by doc special tokens.
    print("Reading dataset from file")
    with open(input_path, 'r', encoding='utf-8') as f:
        file_contents = f.read()
    print("Finished reading dataset from file")
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    print("Splitting documents by special tokens")
    documents = re.split(pattern, file_contents)
    print(f"Number of documents: {len(documents)}")

    # 2. For each document, apply the regex to break the document into words
    words = Counter()
    if num_workers and num_workers > 1:
        batch_size = max(1, len(documents) // num_workers)
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for wc in executor.map(word_count_from_document_batch, document_batches):
                words.update(wc)
    else:
        for document in documents:
            words.update(word_count_from_document(document))
    return words


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    reverse_vocab: dict[bytes, int] = {}
    merges: list[tuple[bytes, bytes]] = []

    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
        reverse_vocab[bytes([token_id])] = token_id
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    # initialize this words dict based on the input_path file
    start = time.time()
    words = populate_words(input_path, special_tokens, reverse_vocab, num_workers=num_workers)
    end = time.time()
    print(f"Time taken to populate words: {end - start} seconds")

    with tqdm(total=max(0, vocab_size - len(vocab)), desc="Training merges") as pbar:
        # build up token_pair_count dictionary
        token_pair_count: dict[tuple[int, int], int] = defaultdict(int)
        # mapping from token pair to words that contain that token pair
        words_with_token_pair: defaultdict[tuple[int, int], set] = defaultdict[Any, set](set)
        for word in words:
            if len(word) == 1:
                continue
            for idx, token_id in enumerate(word):
                if idx < len(word)-1:
                    token_pair_count[word[idx], word[idx+1]] += words[word]
                    words_with_token_pair[word[idx], word[idx+1]].add(word)
        while len(vocab) < vocab_size:
            merge_candidate = max(token_pair_count, key=lambda k: (token_pair_count[k], (vocab[k[0]], vocab[k[1]])))
            new_token_id = len(vocab)
            vocab[new_token_id] = vocab[merge_candidate[0]] + vocab[merge_candidate[1]]
            merges.append((vocab[merge_candidate[0]], vocab[merge_candidate[1]]))

            # Recompute pair counts as a diff between the old word's pairs and
            # the new word's pairs. This naturally handles overlapping (e.g.
            # merging (b,b) in (b,b,b)) and adjacent-match (e.g. merging (t,h)
            # in (t,h,t,h,e)) edge cases that positional bookkeeping gets
            # wrong.
            words_to_update = list(words_with_token_pair[merge_candidate[0], merge_candidate[1]])
            for word in words_to_update:
                new_word = replace_pair(word, (merge_candidate[0], merge_candidate[1]), new_token_id)
                count = words[word]

                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    token_pair_count[pair] -= count
                    if token_pair_count[pair] <= 0:
                        del token_pair_count[pair]
                    words_with_token_pair[pair].discard(word)

                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    token_pair_count[pair] += count
                    words_with_token_pair[pair].add(new_word)

                words[new_word] = count
                del words[word]

            token_pair_count.pop((merge_candidate[0], merge_candidate[1]), None)
            words_with_token_pair.pop((merge_candidate[0], merge_candidate[1]), None)
            pbar.update(1)
    return vocab, merges


def serialize_vocab_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_prefix: str) -> tuple[str, str]:
    with open(output_prefix.with_name(output_prefix.name + "-vocab.txt"), 'w', encoding='utf-8') as f:
        for token_id, token in vocab.items():
            f.write(f"{token_id},{token}\n")
    with open(output_prefix.with_name(output_prefix.name + "-merges.txt"), 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(f"{merge[0]}\n")
            f.write(f"{merge[1]}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--output_folder", type=str, default="outputs")
    args = p.parse_args()

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=['<|endoftext|>'],
        num_workers=args.num_workers,
    )
    end_time = time.time()
    print(f"Time taken to train BPE: {(end_time - start_time)/60.0} minutes")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_prefix = output_folder / Path(args.input_path).stem
    serialize_vocab_merges(vocab, merges, output_prefix)
