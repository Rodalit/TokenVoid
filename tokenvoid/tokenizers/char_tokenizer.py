import json
from .base_tokenizer import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    def train(self, text: str) -> None:
        """
        Trains the tokenizer on the provided text.

        Args:
            text (str): Text for training
        """
        self._init_special_tokens()

        chars = sorted(set(text))

        for idx, token in enumerate(chars, start=len(self.vocab)):
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

    def tokenize(self, text: str) -> list:
        return list(text)
