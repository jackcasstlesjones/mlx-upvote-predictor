import re


def tokenize(text):
    """
    Tokenize text by:
    1. Converting to lowercase
    2. Removing non-alphabetic characters
    3. Splitting on whitespace

    Args:
        text (str): Input text to tokenize

    Returns:
        list: List of tokens
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip().split()


def stream_tokens(path):
    """
    Stream tokens from a text file line by line to keep memory usage low.

    Args:
        path (str): Path to text file

    Yields:
        list: List of tokens for each line
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield tokenize(line)


def load_and_tokenize_text8(text8_data):
    """
    Tokenize text8 data from the Hugging Face dataset

    Args:
        text8_data: Text8 dataset from Hugging Face

    Returns:
        list: List of tokens
    """
    text = text8_data["train"]["text"][0]
    return tokenize(text)

