#!/usr/bin/env python3
"""Generate ground-truth BPE tokenizations for test_tokenizer.cpp verification.

Usage:
    python scripts/verify_tokenizer.py raw_weights/

Loads vocab.json + merges.txt from the given directory and tokenizes the same
test strings used in test_tokenizer.cpp. Prints token IDs for comparison.

This uses a standalone CLIP-style SimpleTokenizer implementation (no deps
beyond the standard library + json).
"""

import json
import os
import re
import sys
from functools import lru_cache


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class SimpleTokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        # Skip header if present
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        merges = [tuple(line.split()) for line in lines if line.strip()]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

        # Use 'regex' module if available (supports \p{L}), else fall back to ASCII
        try:
            import regex
            self.pat = regex.compile(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                regex.IGNORECASE,
            )
        except ImportError:
            # ASCII fallback — sufficient for English test strings
            self.pat = re.compile(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s a-zA-Z0-9]+""",
                re.IGNORECASE,
            )

        self.sot_token = self.encoder.get("<|startoftext|>", 49406)
        self.eot_token = self.encoder.get("<|endoftext|>", 49407)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text, context_length=32):
        # Clean + lowercase
        text = re.sub(r"\s+", " ", text).strip().lower()

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(encoded).split(" ")
            )

        # Build result: [SOT, tokens..., EOT, 0, 0, ...]
        result = [self.sot_token] + bpe_tokens + [self.eot_token]
        if len(result) > context_length:
            result = result[:context_length]
            result[-1] = self.eot_token
        result += [0] * (context_length - len(result))
        return result


TEST_STRINGS = [
    "a",
    "hello world",
    "yellow school bus",
    "a photo of a cat",
    "don't stop",
    "SAM 3",
    "123",
    "",
    "a person riding a bicycle on a sunny day in the park",
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tokenizer_dir>", file=sys.stderr)
        sys.exit(1)

    tok_dir = sys.argv[1]
    vocab_path = os.path.join(tok_dir, "vocab.json")
    merges_path = os.path.join(tok_dir, "merges.txt")

    tokenizer = SimpleTokenizer(vocab_path, merges_path)
    print(f"Loaded {len(tokenizer.encoder)} vocab entries, "
          f"{len(tokenizer.bpe_ranks)} merges")

    print("\n// Expected token IDs (paste into test_tokenizer.cpp if needed):")
    for i, text in enumerate(TEST_STRINGS):
        tokens = tokenizer.encode(text, context_length=32)
        # Print compact
        non_zero = [t for t in tokens if t != 0]
        pad_count = tokens.count(0)
        print(f'// [{i}] "{text}"')
        print(f"//   → {non_zero} + {pad_count} zeros")
        print(f"//   full: {tokens}")
        print()


if __name__ == "__main__":
    main()
