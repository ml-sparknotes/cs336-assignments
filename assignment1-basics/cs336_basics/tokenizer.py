from copy import deepcopy
from typing import Any
import ast
import regex as re


from collections.abc import Iterable
from cs336_basics.bpe_workers import PAT


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key=len, reverse=True)
        self.reverse_vocab = {value: key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # vocab should be a dict loaded from a file like @tinystoriesv2-gpt4-vocab.txt and merges is like @tinystoriesv2-gpt4-merges.txt
        with open(vocab_filepath, "r") as f:
            vocab = dict[int, Any]((int(k), eval(v)) for k, v in (line.strip().split(',', 1) for line in f))
        with open(merges_filepath, "r") as f:
            merge_lines = [line.strip() for line in f if line.strip()]
            merge_bytes = [ast.literal_eval(line) for line in merge_lines]
            merges = [(merge_bytes[i], merge_bytes[i+1]) for i in range(0, len(merge_bytes), 2)]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = self._pretokenize_chunks(self._sp_token_chunk_text(text))
        token_ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                token_ids.append(self.reverse_vocab[chunk.encode("utf-8")])
            else:
                token_ids.extend(self._chunk_to_token_IDs(chunk.encode("utf-8")))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pretokenized_buffer = []
        token_ids = []
        stream_done = False
        yielded = []
        while True:
            # while there are unyielded token ids, yield 'em
            if len(token_ids) > 0:
                token_id = token_ids[0]
                token_ids = token_ids[1:]
                yield token_id

            # fill up pretokenized buffer if its not full
            while len(pretokenized_buffer) < 2:
                try:
                    next_str = next(iterable)
                    string_to_pretok = (pretokenized_buffer[-1] if len(pretokenized_buffer) > 0 else '') + next_str
                    pretokenized_buffer = pretokenized_buffer[:-1] + self._pretokenize_chunks(self._sp_token_chunk_text(string_to_pretok))
                except StopIteration:
                    stream_done = True
                    break

            # extend token ids with newly loaded text
            if stream_done:
                for string in pretokenized_buffer:
                    if string in self.special_tokens:
                        token_ids.append(self.reverse_vocab[string.encode("utf-8")])
                    else:
                        token_ids.extend(self._chunk_to_token_IDs(string.encode("utf-8")))
                break
            else:
                for string in pretokenized_buffer[:-1]:
                    if string in self.special_tokens:
                        token_ids.append(self.reverse_vocab[string.encode("utf-8")])
                    else:
                        token_ids.extend(self._chunk_to_token_IDs(string.encode("utf-8")))
                pretokenized_buffer = pretokenized_buffer[-1:]

        # yeild remaining
        for token_id in token_ids:
            yield token_id
        return

    def decode(self, ids: list[int]) -> str:
        all_bytes = b''
        for token_id in ids:
            all_bytes += (self.vocab[token_id])
        return all_bytes.decode("utf-8", errors="replace")

    def _sp_token_chunk_text(self, text: str):
        if len(self.special_tokens) == 0:
            return [text]
        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        return re.split(f"({pattern})", text)

    def _pretokenize_chunks(self, chunk: list[str]) -> list[str]:
        pretokenized_chunks = []
        for entry in chunk:
            if entry in self.special_tokens:
                pretokenized_chunks.append(entry)
            else:
                pretokenized_chunks.extend(re.findall(PAT, entry))
        return pretokenized_chunks

    def _chunk_to_token_IDs(self, chunk: bytes):        
        # build up an index of bytes and starting positions
        index = {}
        for idx, byte in enumerate(chunk):
            key = bytes([byte])
            if key in index:
                index[key].append(idx)
            else:
                index[key] = [idx]

        # go through merges array and update token_IDs
        for a, b in self.merges:
            next_idx = -1
            if a not in index:
                continue
            for idx in deepcopy(index[a]):
                if idx < next_idx:
                    continue
                total_len = len(a) + len(b)
                if chunk[idx:idx+total_len] == a + b and b in index and (idx + len(a)) in index[b]:
                    self._merge(a, b, idx, index)
                    next_idx = idx + len(a+b)
        # convert to token IDs and return
        tokens = []
        for byte_content in index:
            for pos in index[byte_content]:
                tokens.append((pos, byte_content))
        tokens.sort()
        token_IDs = []
        for _, item in tokens:
            token_IDs.append(self.reverse_vocab[item])
        return token_IDs

    def _merge(self, a, b, start_idx, index):
        index[a].remove(start_idx)
        index[b].remove(start_idx + len(a))
        if a+b in index:
            index[a+b].append(start_idx)
        else:
            index[a+b] = [start_idx]


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/TinyStoriesV2-GPT4-vocab.txt",
        merges_filepath="data/TinyStoriesV2-GPT4-merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    input = "Héllò hôw <|endoftext|><|endoftext|> are ü? "
    tokens = tokenizer.encode(input)
    output = tokenizer.decode(tokens)
    assert input == output

    # input_txt = "Hi Poorva, I heard you're really sexy when you get naked."
    # gen = (x for x in re.split(r'(\s+)', input_txt))
    # token_ids = []
    # for tok_id in tokenizer.encode_iterable(gen):
    #     token_ids.append(tok_id)
    # output = tokenizer.decode(token_ids)
    # print(input_txt)

    # assert input_txt == output
