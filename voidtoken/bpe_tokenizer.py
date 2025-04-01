import json
from collections import Counter
from base_tokenizer import BaseTokenizer

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 1000, special_tokens: dict = {"unk_token": "[UNK]"}):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.unk_token = self.special_tokens["unk_token"]
        self.vocab = {}
        self.inv_vocab = {}
        self.merge_vocab = {}

    def _init_special_tokens(self, text):
        """
        Initializes the base vocabulary (vocab) using the unique characters from the provided text.
        Then, it adds special tokens to the vocabulary by assigning them new indices.
        
        Arguments:
            text (str): The text from which the base vocabulary, consisting of unique characters, is derived.
        
        Notes:
            - Each character in the text is assigned an index based on its order in the set(list(text)).
            - Special tokens (defined in self.special_tokens) are added to the vocabulary with new indices.
        """
        self.vocab = {token: idx for idx, token in enumerate(set(list(text)))}

        for token in self.special_tokens:
            self.vocab[self.special_tokens[token]] = len(self.vocab)

    def _get_stats(self, tokens):
        """
        Calculates the frequency of occurrence for each consecutive pair of tokens (bigram) in the list.
        
        Arguments:
            tokens (list): The list of tokens to analyze.
        
        Returns:
            Counter: A collections.Counter object containing the count of each bigram.
        
        Notes:
            - Uses the zip function to create pairs (tokens[i], tokens[i+1]) for the entire list.
        """
        vocab = Counter()
        for pair in zip(tokens[:-1], tokens[1:]):
            vocab[pair] += 1
        return vocab

    def _merge(self, tokens, pair, merge_pair):
        """
        Merges all occurrences of a specified pair of tokens in the token list into a single merged token.
        
        Arguments:
            tokens (list): The original list of tokens.
            pair (tuple): The pair of tokens that need to be merged.
            merge_pair: The token that will replace the specified pair.
        
        Returns:
            list: A new list of tokens where every occurrence of the specified pair is replaced by the merged token.
        
        Notes:
            - The function iterates through the token list, checking adjacent tokens.
            - If the specified pair is found, it is replaced with merge_pair and the next token is skipped; otherwise, the current token is appended as-is.
        """
        newids = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                newids.append(merge_pair)
                i += 2
            else:
                newids.append(tokens[i])
                i += 1
        return newids

    def train(self, text):
        """
        Trains the model on a given text by iteratively merging the most frequent pairs of tokens until the target
        vocabulary size is reached or no more pairs can be merged.

        Steps:
          1. Initializes the vocabulary with unique characters and special tokens using _init_special_tokens.
          2. Converts the text into a list of tokens (initially individual characters).
          3. Iteratively, for as many iterations as needed to reach the target vocabulary size:
             - Computes token pair frequencies with _get_stats.
             - If no token pairs are found, terminates the loop.
             - Identifies the most frequent pair of tokens.
             - Creates a new token by merging the best pair.
             - Updates the vocabulary with the new merged token.
             - Stores the mapping from the original token pair to the merged token.
             - Merges the tokens in the token list using _merge.
          4. Creates an inverse vocabulary (inv_vocab) mapping indices back to tokens.

        Arguments:
          text (str): The input text on which to train the model.

        Notes:
          - The merging procedure is similar to algorithms used in subword tokenization methods like Byte Pair Encoding (BPE).
          - self.vocab_size should be predefined and represents the target size of the vocabulary.
        """
        self._init_special_tokens(text)

        tokens = list(text)

        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_stats(tokens)

            if not pairs:
                break

            best_pair = max(pairs, key=lambda x: pairs[x])
            merge_pair = "".join(best_pair)
            self.vocab[merge_pair] = len(self.vocab)
            self.merge_vocab[best_pair] = merge_pair

            tokens = self._merge(tokens, best_pair, merge_pair)

        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, text):
        """
        Tokenizes an input text by iteratively merging tokens according to the merge vocabulary.

        Steps:
          1. Converts the text into a list of tokens (initially each character is a token).
          2. For as many iterations as there are merge operations:
             - Iterates through the token list and checks if any pair of adjacent tokens exists in the merge vocabulary.
             - If the pair is found, it replaces the pair with the corresponding merged token and skips the next token.
             - Otherwise, the current token is retained.
             - The token list is updated after each iteration.
          3. Returns the final list of tokens after all possible merge operations have been applied.

        Arguments:
          text (str): The input text that needs to be tokenized.

        Returns:
          list: A tokenized representation of the input text, where merge operations have been applied iteratively.
        
        Notes:
          - This method applies the same number of iterations as there are pairs in the merge vocabulary, ensuring that all merge rules are considered.
          - The merge rules applied here should have been previously learned during training.
        """
        tokens = list(text)
        for _ in range(len(self.merge_vocab)):
            newids = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merge_vocab:
                    newids.append(self.merge_vocab[(tokens[i], tokens[i + 1])])
                    i += 2
                else:
                    newids.append(tokens[i])
                    i += 1
            tokens = newids
        return newids

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