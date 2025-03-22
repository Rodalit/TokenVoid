import json

class CharacterTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def train(self, text: str) -> None:
        """
        Trains the tokenizer on the provided text.

        Args:
            text (str): Text for training
        """

        chars = sorted(set(text))

        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}

    def tokenize(self, text: str) -> list:
        return list(text)

    def encode(self, text: str) -> list:
        """
        Sequence Encoding Function.

        Args:
            text (str): Text to be encoded
        """
        if not text:
            raise ValueError("The field cannot be empty")

        return [self.char2idx[char] for char in list(text)]

    def decode(self, tokens: list) -> str:
        """
        Decoding a sequence of tokens back to text

        Args:
            tokens (list): Token sequence
        """
        if not tokens:
            raise ValueError("The field cannot be empty")

        return "".join(self.idx2char[index] for index in tokens)

    def save(self, filepath: str) -> None:
        """
        Function for saving token dictionaries

        Args:
            filepath (str): The path where you need to save the file, do not forget to specify the file name and its format (json)
        """

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.char2idx, file)

    def load(self, filepath: str) -> None:
        """
        Function for loading a token dictionary

        Args:
            filepath (str): Path to the file with dictionaries
        """

        with open(filepath, "r", encoding='utf-8') as file:
            self.char2idx = json.load(file)
            self.idx2char = {i: c for c, i in self.char2idx.items()}