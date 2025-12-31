import re
from typing import List, Optional
from pydantic import BaseModel


class TextChunk(BaseModel):
    """Model for text chunks"""
    id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: dict = {}


class TextUtils:
    """Utility class for text processing operations"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by removing extra whitespace and standardizing formatting.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 1000, overlap: int = 100) -> List[TextChunk]:
        """
        Split text into chunks of approximately max_tokens with overlap.

        Args:
            text: Input text to chunk
            max_tokens: Maximum number of tokens per chunk
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks with metadata
        """
        # For now, we'll use a simple approach based on character count
        # In a real implementation, this would use proper tokenization
        words = text.split()
        chunks = []
        chunk_id = 0

        i = 0
        while i < len(words):
            # Start with the first word
            chunk_words = []
            current_length = 0

            # Add words until we reach the max token count
            while i < len(words) and current_length < max_tokens:
                chunk_words.append(words[i])
                current_length += 1
                i += 1

            # Create chunk text
            chunk_text = ' '.join(chunk_words)

            # Create chunk with metadata
            chunk = TextChunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text,
                start_pos=len(' '.join(words[:i-len(chunk_words)])),
                end_pos=len(' '.join(words[:i])),
                metadata={
                    "token_count": len(chunk_words),
                    "original_position": i - len(chunk_words)
                }
            )

            chunks.append(chunk)
            chunk_id += 1

            # Move back by overlap amount if there are more words
            if i < len(words) and overlap > 0:
                i = max(i - overlap, i - len(chunk_words))

        return chunks

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text (simplified approximation).

        Args:
            text: Input text to count tokens for
            model: Model name for tokenization (not used in this simple version)

        Returns:
            Approximate token count
        """
        # This is a simple approximation - in a real implementation,
        # we'd use proper tokenizers like tiktoken
        if not text:
            return 0
        # Rough approximation: 1 token ~ 4 characters
        return len(text) // 4

    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Extract sentences from text.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting using common sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Clean up and remove empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters and normalizing.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\n\r]', ' ', text)
        # Normalize whitespace
        text = TextUtils.normalize_text(text)
        return text