from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import regex as re
from tqdm import tqdm
import time
from collections import Counter

try:
    from cs336_basics.bpe_workers import word_count_from_document, word_count_from_document_batch
except ModuleNotFoundError:
    from bpe_workers import word_count_from_document, word_count_from_document_batch


def contains_subtuple(t, sub):
    """
    This function checks if a tuple contains a sub-tuple.
    Args:
        t: The tuple to check.
        sub: The sub-tuple to check for.
    Returns:
        True if the tuple contains the sub-tuple, False otherwise.
    """
    n, m = len(t), len(sub)
    return any(t[i:i+m] == sub for i in range(n - m + 1))


def replace_pair(t: tuple[int, ...], pair: tuple[int, int], new_val: int) -> tuple[int, ...]:
    """
    This function replaces a pair of tokens in a tuple with a new token.
    Args:
        t: The tuple to replace the pair in.
        pair: The pair of tokens to replace.
        new_val: The new token to replace the pair with.
    Returns:
        A new tuple with the pair replaced by the new token.
    """
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
        while len(vocab) < vocab_size:
            token_pair_count: dict[tuple[int, int], int] = defaultdict(int)
            # find the counts of all consecutive token pairs
            for word in words:
                if len(word) == 1:
                    continue
                for idx, token_id in enumerate(word):
                    if idx < len(word)-1:
                        token_pair_count[word[idx], word[idx+1]] += words[word]
            merge_candidate = max(token_pair_count, key=lambda k: (token_pair_count[k], (vocab[k[0]], vocab[k[1]])))
            new_token_id = len(vocab)
            vocab[new_token_id] = vocab[merge_candidate[0]] + vocab[merge_candidate[1]]
            merges.append((vocab[merge_candidate[0]], vocab[merge_candidate[1]]))
            # update words dict so that it takes the latest merge into account
            words_to_update = []
            for word in words:
                if contains_subtuple(word, (merge_candidate[0], merge_candidate[1])):
                    words_to_update.append(word)
            for word in words_to_update:
                new_word = replace_pair(word, (merge_candidate[0], merge_candidate[1]), new_token_id)
                words[new_word] = words[word]
                del words[word]
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
    data_dir = Path(__file__).resolve().parent / "data"
    vocab, merges = train_bpe(
        str(data_dir / "TinyStoriesV2-GPT4-train.txt"),
        1000,
        ['<|endoftext|>'],
        num_workers=20,
    )

    serialize_vocab_merges(vocab, merges, data_dir / "TinyStoriesV2-GPT4")
