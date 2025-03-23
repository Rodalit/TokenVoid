import re
import json
from collections import Counter

class SentencesTokenizer:
    def __init__(self, vocab_size: int = 10_000, special_tokens: list = ["<|unk|>"]):
        """
        Initializes the SentencesTokenizer with a specified vocabulary size and special tokens.

        Parameters:
        vocab_size (int): The size of the vocabulary.
        special_tokens (list): A list of special tokens to be included in the vocabulary.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.sen2idx = {}
        self.idx2sen = {}

    def tokenize(self, text: str) -> list:
        """
        Tokenizes the input text into sentences based on punctuation marks.

        Parameters:
        text (str): The input text to be tokenized.

        Returns:
        list: A list of sentences.
        """
        return re.split(r'(?<=[.!?])\s+', text)

    def train(self, text: str) -> None:
        """
        Trains the tokenizer on the input text to build the vocabulary.

        Parameters:
        text (str): The input text to train on.
        """
        if self.special_tokens:
            for token in self.special_tokens:
                self.sen2idx[token] = len(self.sen2idx)
                self.idx2sen[len(self.idx2sen)] = token

        sentences = self.tokenize(text)

        freq = Counter(sentences)

        most_popular_sentences = [sentences for sentences, _ in freq.most_common(self.vocab_size)]

        for idx, sentence in enumerate(most_popular_sentences, start=len(self.sen2idx)):
            self.sen2idx[sentence] = idx
            self.idx2sen[idx] = sentence

    def encode(self, text: str) -> list:
        """
        Encodes the input text into a list of token indices.

        Parameters:
        text (str): The input text to be encoded.

        Returns:
        list: A list of token indices.
        """
        return [self.sen2idx.get(token, self.sen2idx["<|unk|>"]) for token in self.tokenize(text)]

    def decode(self, tokens: list) -> str:
        """
        Decodes a list of token indices back into a string of sentences.

        Parameters:
        tokens (list): A list of token indices to be decoded.

        Returns:
        str: The decoded string of sentences.
        """
        return " ".join(self.idx2sen[token] for token in tokens)

    def save(self, filepath):
        """
        Saves the sentence-to-index mapping to a file.

        Parameters:
        filepath (str): The file path to save the mapping.
        """
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.sen2idx, file)

    def load(self, filepath):
        """
        Loads the sentence-to-index mapping from a file.

        Parameters:
        filepath (str): The file path to load the mapping from.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            self.sen2idx = json.load(file)
            self.idx2sen = {idx: sentence for sentence, idx in self.sen2idx.items()}
