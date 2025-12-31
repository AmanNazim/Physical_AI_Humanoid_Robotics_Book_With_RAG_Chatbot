"""
Dynamic chunking engine with 800-1200 token range
"""
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import hashlib
import logging
from .config import CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: str
    content: str
    token_count: int
    character_start: int
    character_end: int
    token_start: int
    token_end: int
    parent_chunk_id: Optional[str] = None
    overlap_type: str = "none"  # "before", "after", "none"
    document_reference: Optional[str] = None
    page_reference: Optional[int] = None
    section_title: Optional[str] = None
    content_hash: Optional[str] = None
    overlap_before: Optional[str] = None
    overlap_after: Optional[str] = None
    overlap_size: Optional[int] = None


class ChunkingEngine:
    """Class for dynamic chunking with 800-1200 token range"""

    def __init__(self):
        self.min_chunk_size = CHUNK_SIZE_MIN
        self.max_chunk_size = CHUNK_SIZE_MAX
        self.overlap_size = CHUNK_OVERLAP
        self.target_chunk_size = (self.min_chunk_size + self.max_chunk_size) // 2  # 1000

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count by counting words (rough approximation)
        In practice, you'd use a proper tokenizer like tiktoken
        """
        # For now, use a simple word-based estimation
        # In a real implementation, use a proper tokenizer like tiktoken
        words = re.findall(r'\b\w+\b', text)
        # Average English word is ~1.3 tokens, so we'll use 1.2 for safety
        return int(len(words) * 1.2)

    def create_chunk(self, content: str, doc_ref: str = None, char_start: int = 0, char_end: int = 0,
                    token_start: int = 0, token_end: int = 0) -> Chunk:
        """Create a chunk with proper metadata"""
        chunk_id = hashlib.sha256(f"{content}{char_start}{char_end}".encode()).hexdigest()[:16]

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            token_count=self.estimate_token_count(content),
            character_start=char_start,
            character_end=char_end,
            token_start=token_start,
            token_end=token_end,
            document_reference=doc_ref,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )

    def chunk_by_tokens(self, text: str, document_reference: Optional[str] = None) -> List[Chunk]:
        """
        Split text into chunks based on token count with overlap
        """
        if not text:
            return []

        # Estimate total tokens in text
        total_tokens = self.estimate_token_count(text)
        chunks = []

        # Split text into sentences to use as boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        current_chunk = ""
        current_start = 0
        current_token_count = 0

        for i, sentence in enumerate(sentences):
            # Estimate tokens in this sentence
            sentence_tokens = self.estimate_token_count(sentence)

            # If adding this sentence would exceed max chunk size
            if current_token_count + sentence_tokens > self.max_chunk_size and current_chunk:
                # Create a chunk with the current content
                chunk = self.create_chunk(
                    content=current_chunk.strip(),
                    doc_ref=document_reference,
                    char_start=current_start,
                    char_end=current_start + len(current_chunk),
                    token_start=0,  # Would need actual tokenization to get precise counts
                    token_end=current_token_count
                )
                chunks.append(chunk)

                # Start a new chunk with overlap
                if len(chunks) > 0:
                    # Add overlap from previous chunk
                    prev_chunk_content = chunks[-1].content
                    overlap_start_idx = max(0, len(prev_chunk_content) - 100)  # Approximate overlap
                    overlap_content = prev_chunk_content[overlap_start_idx:]

                    current_chunk = overlap_content + " " + sentence
                    current_start = len(" ".join([c.content for c in chunks[:-1]] + [overlap_content]))
                else:
                    current_chunk = sentence
                    current_start = 0

                current_token_count = self.estimate_token_count(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

                current_token_count += sentence_tokens

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunk = self.create_chunk(
                content=current_chunk.strip(),
                doc_ref=document_reference,
                char_start=current_start,
                char_end=current_start + len(current_chunk),
                token_start=0,  # Would need actual tokenization
                token_end=current_token_count
            )
            chunks.append(chunk)

        # Apply overlap logic after creating all chunks
        chunks = self.apply_overlap(chunks)

        return chunks

    def apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply overlap logic between chunks"""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            new_chunk = chunk

            # Add overlap from previous chunk if exists
            if i > 0:
                prev_chunk = chunks[i-1]
                # Extract last part of previous chunk as overlap
                prev_content_words = prev_chunk.content.split()
                overlap_words = prev_content_words[-max(1, int(self.overlap_size * 0.1)):]  # Approximate
                overlap_content = " ".join(overlap_words)

                new_chunk.overlap_before = overlap_content
                new_chunk.overlap_size = len(overlap_words)

            # Add overlap to next chunk if exists
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                # Extract first part of next chunk as overlap
                next_content_words = next_chunk.content.split()
                overlap_words = next_content_words[:max(1, int(self.overlap_size * 0.1))]  # Approximate
                overlap_content = " ".join(overlap_words)

                new_chunk.overlap_after = overlap_content

            overlapped_chunks.append(new_chunk)

        return overlapped_chunks

    def validate_chunk_size(self, chunks: List[Chunk]) -> List[str]:
        """Validate that all chunks are within size constraints"""
        errors = []
        for i, chunk in enumerate(chunks):
            if chunk.token_count < self.min_chunk_size:
                errors.append(f"Chunk {i} ({chunk.chunk_id}) has {chunk.token_count} tokens, below minimum of {self.min_chunk_size}")
            if chunk.token_count > self.max_chunk_size:
                errors.append(f"Chunk {i} ({chunk.chunk_id}) has {chunk.token_count} tokens, above maximum of {self.max_chunk_size}")
        return errors


class ChunkProcessor:
    """Class for chunking operations"""

    def __init__(self):
        self.engine = ChunkingEngine()

    def chunk_text(self, text: str, document_reference: Optional[str] = None) -> List[Chunk]:
        """Split text into chunks according to specifications"""
        # Validate input
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Create chunks
        chunks = self.engine.chunk_by_tokens(text, document_reference)

        # Validate chunks
        validation_errors = self.engine.validate_chunk_size(chunks)
        if validation_errors:
            logger.warning(f"Chunk validation errors: {validation_errors}")

        return chunks