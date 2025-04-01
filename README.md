# VoidToken
The library is a tool whose goal is to simplify LLM development and text analysis. The library provides different kinds of tokenizers; all tokenizers are very flexible and customizable. The library is currently under development, but already provides tokenization tools.

Example code:

```python
import voidtoken

example = "Hello world!"

tokenizer = voidtoken.CharacterTokenizer()

tokens = tokenizer.tokenize(example)

print(tokens) # ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']
``` 

The library already has:

    - CharacterTokenizer - divides text by letters.
    - WordTokenizer - divides text by words.
    - BPETokenizer - compresses text by combining the most common character sequences into new tokens.

Still to be done:

    - Tidy up the code.
    - Collect data to train tokenizers.
    - Add pre-trained dictionaries with tokenizers.
    - Add more tokenizers.

NOTE: I am not an expert in NLP, but I am actively exploring this field. I will be glad to help in developing and improving my library, as well as any advice and recommendations.