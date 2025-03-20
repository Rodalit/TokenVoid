import re
import json
from collections import Counter

class WordTokenizer:
	def __init__(self, vocab_size: int = 10_000):
		self.word2idx = {"<|PAD|>": 0, "<|SOS|>": 1, "<|EOS|>": 2, "<|UNK|>": 3}
		self.idx2word = {}
		self.vocab_size = vocab_size
		self.pattern = r'([,.:;?_!"()\']|--|\s)'

	def train(self, text: str) -> None:
		"""
		Train the tokenizer on the corpus
		
		Args:
			text (str): Text on which the tokenizer will be trained
		"""
		tokens = re.split(self.pattern, text)
		
		freq = Counter(tokens)

		most_popular_words = [word for word, _ in freq.most_common(self.vocab_size-4)]

		for index, token in enumerate(most_popular_words):
			self.word2idx[token] = index + 4

		self.idx2word = {index: token for token, index in self.word2idx.items()}

	def encode(self, text: str, add_special_tokens: bool = False) -> list:
		"""
		Encoding a sequence into tokens

		Args:
			text (str): Text to be encoded
			add_special_tokens (bool): Responsible for adding special tokens (<|SOS|>, <|EOS|>, etc.)
		"""
		if text is None:
			raise ValueError("The value must not be empty")

		if add_special_tokens:
			text = f"<|SOS|> {text} <|EOS|>"
		tokens = re.split(self.pattern, text)
		return [self.word2idx.get(token, self.word2idx["<|UNK|>"]) for token in tokens]

	def decode(self, tokens: list) -> str:
		"""
		Decoding a tokens into sequence

		Args:
			tokens (list): Token sequence
		"""
		if tokens is None:
			raise ValueError("The value must not be empty")

		return "".join([self.idx2word[token] for token in tokens])

	def save(self, filepath: str) -> None:
		"""
		Function for saving the token dictionary

		Args:
			filepath (str): The path where the file should be saved
		"""
		if filepath is None:
			raise ValueError("The value must not be empty")

		with open(filepath, "w", encoding="utf-8") as file:
			json.dump(self.word2idx, file)

	def load(self, filepath: str) -> None:
		"""
		Function for loading the dictionary

		Args:
			filepath (str): Path to the file that needs to be loaded
		"""
		if filepath is None:
			raise ValueError("The value must not be empty")
		
		with open(filepath, "r", encoding="utf-8") as file:
			self.word2idx = json.load(file)
			self.idx2word = {index: token for token, index in self.word2idx.items()}