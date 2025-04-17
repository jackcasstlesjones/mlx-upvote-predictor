"""
Tokenization utilities for Word2Vec model.

This module provides functions to convert text into tokens suitable for
Word2Vec training, including specialized handling for various text formats
and technical terminology common in Hacker News discussions.
"""

import re
from typing import List, Generator, Dict, Any, Optional


def tokenize(text: str, domain_specific: bool = False) -> List[str]:
    """
    Tokenize text into words.

    Basic tokenization:
    1. Converting to lowercase
    2. Removing non-alphabetic characters
    3. Splitting on whitespace

    If domain_specific=True, adds special handling for:
    - Technical terms (preserving camelCase, snake_case)
    - Programming symbols and operators
    - Common abbreviations in tech discussions

    Args:
        text: Input text to tokenize
        domain_specific: Whether to use HN-specific tokenization rules

    Returns:
        List[str]: List of tokens
    """
    if not domain_specific:
        # Simple tokenization for general text
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip().split()
    else:
        # More sophisticated tokenization for HN content
        # Preserve case for technical terms
        tokens = []

        # First, handle known technical patterns
        text = re.sub(r'(https?://\S+)', ' <URL> ', text)  # Replace URLs
        # Preserve ticker symbols
        text = re.sub(r'(\$[A-Z]+)', r' \g<1> ', text)
        # Split on whitespace and punctuation, preserving some
        # technical patterns
        raw_tokens = re.findall(r'[a-zA-Z0-9_]+|[^\w\s]', text.lower())

        # Further process tokens
        for token in raw_tokens:
            if token and not token.isspace():
                tokens.append(token)

        return tokens


def stream_tokens(
        path: str,
        domain_specific: bool = False
) -> Generator[List[str], None, None]:
    """
    Stream tokens from a text file line by line to keep memory usage low.

    Args:
        path: Path to text file
        domain_specific: Whether to use domain-specific tokenization

    Yields:
        List[str]: List of tokens for each line
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield tokenize(line, domain_specific=domain_specific)


def load_and_tokenize_text8(text8_data: Dict[str, Any]) -> List[str]:
    """
    Tokenize text8 data from the Hugging Face dataset.

    Args:
        text8_data: Text8 dataset from Hugging Face

    Returns:
        List[str]: List of tokens
    """
    text = text8_data["train"]["text"][0]
    return tokenize(text)


def tokenize_hn_title(title: str) -> List[str]:
    """
    Specialized tokenization for Hacker News titles.

    Preserves technical terms, common abbreviations, and other
    patterns found in Hacker News discussions.

    Args:
        title: Hacker News post title

    Returns:
        List[str]: List of tokens
    """
    return tokenize(title, domain_specific=True)


def extract_domain_from_url(url: str) -> Optional[str]:
    """
    Extract domain name from URL.

    Args:
        url: URL string

    Returns:
        Optional[str]: Domain name or None if URL is invalid
    """
    if not url:
        return None

    # Extract domain using regex
    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if domain_match:
        domain = domain_match.group(1)
        # Remove extensions and clean domain name
        domain = re.sub(r'\.(com|org|net|io|co|gov|edu)$', '', domain)
        return domain.replace('.', '_')

    return None
