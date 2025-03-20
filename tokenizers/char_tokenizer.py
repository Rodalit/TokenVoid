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
			json.dump({
					"char2idx": self.char2idx,
					"idx2char": self.idx2char
				}, file, ensure_ascii=False, indent=2)

	def load(self, filepath: str) -> None:
		"""
		Function for loading a token dictionary

		Args:
			filepath (str): Path to the file with dictionaries
		"""

		with open(filepath, "r", encoding='utf-8') as file:
			loaded_data = json.load(file)

		self.char2idx = loaded_data['char2idx']
		self.idx2char = loaded_data['idx2char']