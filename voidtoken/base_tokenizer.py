import re
import json
from collections import Counter

class BaseTokenizer:
    def __init__(self, 
        vocab_size: int = 10_000, 
        special_tokens: dict = {"pad_token": "<|pad|>", "unk_token": "<|unk|>",
                                "bos_token": "<|bos|>", "eos_token": "<|eos|>"}):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.unk_token = self.special_tokens["unk_token"]
        self.vocab = {}
        self.inv_vocab = {}

    def _init_special_tokens(self) -> None:
        for token in self.special_tokens:
            self.vocab[self.special_tokens[token]] = len(self.vocab)
            self.inv_vocab[len(self.inv_vocab)] = self.special_tokens[token]

    def get_token_id(self, token: str) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            return "Token not found."

    def tokenize(self, text: str) -> None:
        raise NotImplementedError("The tokenize() method must be overridden in the subclass.")

    def train(self, text: str) -> None:
        raise NotImplementedError("The train() method must be overridden in the subclass.")

    def encode(self, text: str, add_special_tokens: bool = False, classification: bool = False):
        if add_special_tokens:
            if "bos_token" in self.special_tokens and "eos_token" in self.special_tokens:
                tokens = [self.special_tokens["bos_token"]] + self.tokenize(text) + [self.special_tokens["eos_token"]]
                return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in self.tokenize(text)]

    def decode(self, tokens: list) -> str:
        return "".join(self.inv_vocab[token] for token in tokens)

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as file:
            self.vocab = json.load(file)
            self.inv_vocab = {index: token for token, index in self.vocab.items()}