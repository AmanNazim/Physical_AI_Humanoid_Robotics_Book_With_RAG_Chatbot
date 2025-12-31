"""
Text preprocessor for text normalization
"""
import re
import unicodedata
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Class for text normalization and preprocessing"""

    def __init__(self):
        # Define regex patterns for various cleaning operations
        self.whitespace_pattern = re.compile(r'\s+')
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.special_char_pattern = re.compile(r'[^\w\s\-.(),!?;:\'"&]')

    def normalize_text(self, text: str) -> str:
        """
        Normalize input text by applying standard normalization rules
        """
        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)

        # 2. Remove control characters except standard whitespace
        text = self.control_char_pattern.sub(' ', text)

        # 3. Remove HTML tags if present
        text = self.html_tag_pattern.sub('', text)

        # 4. Whitespace normalization
        text = self.whitespace_pattern.sub(' ', text).strip()

        return text

    def sanitize_content(self, text: str) -> str:
        """
        Sanitize content to remove potentially harmful elements
        """
        # Remove any potential script tags or other dangerous content
        dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers like onclick, onload, etc.
        ]

        for pattern in dangerous_patterns:
            text = pattern.sub('', text)

        return text

    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate text content for proper format and structure
        """
        errors = []

        # Check for binary data signatures
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Text contains invalid UTF-8 characters")

        # Check for minimum length (if needed)
        if len(text.strip()) == 0:
            errors.append("Text is empty")

        # Check for excessive special characters (potential binary data)
        special_char_ratio = len(self.special_char_pattern.findall(text)) / len(text) if text else 0
        if special_char_ratio > 0.5:  # More than 50% special characters
            errors.append("Text contains excessive special characters, possibly binary data")

        return len(errors) == 0, errors

    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """
        Complete preprocessing pipeline: normalize, sanitize, validate
        Returns: (processed_text, validation_errors)
        """
        # 1. Sanitize content first
        sanitized_text = self.sanitize_content(text)

        # 2. Normalize text
        normalized_text = self.normalize_text(sanitized_text)

        # 3. Validate text
        is_valid, validation_errors = self.validate_text(normalized_text)

        if not is_valid:
            logger.warning(f"Text validation failed with errors: {validation_errors}")

        return normalized_text, validation_errors

    def preprocess_batch(self, texts: List[str]) -> List[Tuple[str, List[str]]]:
        """
        Preprocess a batch of texts
        """
        results = []
        for text in texts:
            processed_text, errors = self.preprocess(text)
            results.append((processed_text, errors))

        return results