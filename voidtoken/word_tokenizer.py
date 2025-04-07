import re
import json
from itertools import chain
from collections import Counter
from .base_tokenizer import BaseTokenizer

class WordTokenizer(BaseTokenizer):
    def train(self, corpus: str) -> None:
        """
        Train the tokenizer on the corpus

        Args:
            corpus (list): Training data set.
        """
        self._init_special_tokens()

        tokens = chain.from_iterable(map(self.tokenize, corpus))

        freq = Counter(tokens)

        most_popular_words = [word for word, _ in freq.most_common(self.vocab_size - len(self.vocab))]

        for index, token in enumerate(most_popular_words, start=len(self.vocab)):
            self.vocab[token] = index
            self.inv_vocab[index] = token

    def tokenize(self, text: str) -> list:
        tokens = [t for t in re.split(r'([,.:;?!"()\']|--|\s+)', text)]
        return tokens