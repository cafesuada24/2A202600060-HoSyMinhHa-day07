from __future__ import annotations

import math
import re
from functools import reduce


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [sent.strip() for sent in sentences]
        return [
            " ".join(sentences[i : i + self.max_sentences_per_chunk])
            for i in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 500,
    ) -> None:
        self.separators = (
            self.DEFAULT_SEPARATORS if separators is None else list(separators)
        )
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        final_chunks = []

        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == '':
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i+1:]

        if separator != '':
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
                continue
            if good_splits:
                final_chunks.extend(self._merge_splits(good_splits, separator))
                good_splits = []
            final_chunks.extend(self._split(s, new_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        merged = []
        current_doc = []
        total_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if current_doc else 0)
            
            if total_len + s_len <= self.chunk_size:
                current_doc.append(s)
                total_len += s_len
            else:
                if current_doc:
                    merged.append(separator.join(current_doc))
                current_doc = [s]
                total_len = len(s)
                
        if current_doc:
            merged.append(separator.join(current_doc))
        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    numerator = _dot(vec_a, vec_b)
    denumerator = _dot(vec_a, vec_a) * _dot(vec_b, vec_b)
    if math.isclose(denumerator, 0, abs_tol=1e-6):
        return 0
    return numerator / denumerator


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        raise NotImplementedError("Implement ChunkingStrategyComparator.compare")
