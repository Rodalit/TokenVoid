import re
import json
from collections import Counter
from .base_tokenizer import BaseTokenizer

class SentencesTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list:
        """
        Tokenizes the input text into sentences based on punctuation marks.

        Parameters:
        text (str): The input text to be tokenized.

        Returns:
        list: A list of sentences.
        """
        return re.split(r'(?<=[.!?])(\s+)', text)

    def train(self, text: str) -> None:
        """
        Trains the tokenizer on the input text to build the vocabulary.

        Parameters:
        text (str): The input text to train on.
        """
        self._init_special_tokens()

        sentences = self.tokenize(text)

        freq = Counter(sentences)

        most_popular_sentences = [sentences for sentences, _ in freq.most_common(self.vocab_size)]

        for idx, sentence in enumerate(most_popular_sentences, start=len(self.vocab)):
            self.vocab[sentence] = idx
            self.inv_vocab[idx] = sentence
