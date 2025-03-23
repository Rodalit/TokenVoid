import json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000, special_tokens: list = ["<|unk|>"]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        self.inv_vocab = {}
        self.merge_vocab = {}

    def _bpe_merge(self, text: list) -> list:
        """Performs one BPE merge based on frequency."""
        if len(text) < 2:
            return text

        pairs = Counter(zip(text, text[1:]))
        if not pairs:
            return text

        most_common_pair, _ = pairs.most_common(1)[0]
        merged_token = "".join(most_common_pair)
        self.merge_vocab[most_common_pair] = merged_token

        new_text = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and (text[i], text[i + 1]) == most_common_pair:
                new_text.append(merged_token)
                i += 2
            else:
                new_text.append(text[i])
                i += 1
        return new_text

    def train(self, text: str) -> None:
        """Trains BPE on the given text."""

        tokens = list(text)
        vocab_set = set(tokens)

        for _ in range(self.vocab_size - len(self.special_tokens)):
            new_tokens = self._bpe_merge(tokens)
            if new_tokens == tokens:
                break
            tokens = new_tokens
            vocab_set.update(tokens)

            if len(vocab_set) >= self.vocab_size:
                break

        # Adding special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[len(self.inv_vocab)] = token

        # Filling the main dictionary
        for idx, token in enumerate(vocab_set, start=len(self.vocab)):
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

    def tokenize_with_bpe(self, text: str) -> list:
        """Tokenizes the text using the saved BPE merges."""
        tokens = list(text)
        for _ in range(len(self.merge_vocab)):  # Use only known merges
            new_tokens = []
            i = 0
            while i < len(tokens):
                merged = None
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merge_vocab:
                    merged = self.merge_vocab[(tokens[i], tokens[i + 1])]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if new_tokens == tokens:
                break
            tokens = new_tokens
        return tokens

    def encode(self, text: str) -> list:
        """Encodes the text into indices using BPE."""
        tokens = self.tokenize_with_bpe(text)
        return [self.vocab.get(token, self.vocab["<|unk|>"]) for token in tokens]

    def decode(self, tokens: list) -> str:
        """Decodes tokens into a sequence."""
        return "".join([self.inv_vocab[token] for token in tokens])

    def save(self, filepath):
        """Save the main dictionary and the merge dictionary."""
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump({
                    "vocab": self.vocab,
                    "merge": self.merge_vocab
                }, file, ensure_ascii=False, indent=2)

    def load(self, filepath):
        """Load the dictionary and the merge dictionary"""
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        self.vocab = loaded_data['vocab']
        self.inv_vocab = {idx: token for token, idx in self.char2idx.items()}
        self.merge_vocab = loaded_data['merge']