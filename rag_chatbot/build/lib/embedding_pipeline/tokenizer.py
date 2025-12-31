"""
Tokenizer module for the Embeddings & Chunking Pipeline.
This module implements tokenization for proper chunking according to specifications.
"""
import tiktoken


def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """
    Get a tokenizer for the specified encoding.

    Args:
        encoding_name: Name of the encoding to use (default: cl100k_base which works well for most text)

    Returns:
        Configured tiktoken encoder
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the given text.

    Args:
        text: Input text to count tokens for
        encoding_name: Name of the encoding to use

    Returns:
        Number of tokens in the text
    """
    encoder = get_tokenizer(encoding_name)
    tokens = encoder.encode(text)
    return len(tokens)


def encode_tokens(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    """
    Encode text into token IDs.

    Args:
        text: Input text to encode
        encoding_name: Name of the encoding to use

    Returns:
        List of token IDs
    """
    encoder = get_tokenizer(encoding_name)
    return encoder.encode(text)


def decode_tokens(token_ids: list[int], encoding_name: str = "cl100k_base") -> str:
    """
    Decode token IDs back to text.

    Args:
        token_ids: List of token IDs to decode
        encoding_name: Name of the encoding to use

    Returns:
        Decoded text
    """
    encoder = get_tokenizer(encoding_name)
    return encoder.decode(token_ids)