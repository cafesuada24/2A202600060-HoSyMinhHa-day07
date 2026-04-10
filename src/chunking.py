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
        """Initializes the FixedSizeChunker.

        Args:
            chunk_size: Maximum number of characters in each chunk.
            overlap: Number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Splits the input text into fixed-size chunks.

           Fixed-size chunking is a foundational technique in RAG. By maintaining
           a consistent chunk size, we ensure that each retrieved segment fits
           into the LLM's context window. The 'overlap' is crucial because it
           prevents semantic loss at boundaries, ensuring that concepts split
           across chunks can still be understood in context.

        Args:
            text: The raw input string to be chunked.

        Returns:
            A list of strings, each being a chunk of text.
        """
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
        """Initializes the SentenceChunker.

        Args:
            max_sentences_per_chunk: Maximum number of sentences allowed in a single chunk.
        """
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        """Splits the text into chunks based on sentence boundaries.

           Sentence-based chunking is more semantically aware than fixed-size
           chunking. By respecting sentence boundaries, we preserve the syntactic
           integrity of the text, which often leads to higher-quality embeddings
           as each chunk represents a complete set of thoughts or assertions.

        Args:
            text: The raw input string.

        Returns:
            A list of strings where each element is a group of sentences.
        """
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
        """Initializes the RecursiveChunker.

        Args:
            separators: A list of separator strings to try in order.
            chunk_size: Target maximum size for each chunk.
        """
        self.separators = (
            self.DEFAULT_SEPARATORS if separators is None else list(separators)
        )
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """Splits text using a recursive strategy.

           Recursive chunking is often considered the 'gold standard' for RAG.
           It attempts to split text at the most logical points (paragraphs,
           then sentences, then words) until the resulting chunks are below
           the target size. This hierarchical approach preserves the document's
           natural structure better than any other method.

        Args:
            text: The raw input string.

        Returns:
            A list of strings optimized for both size and semantic coherence.
        """
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Internal recursive split implementation.

           The recursion works by picking the highest-priority separator that
           exists in the text and splitting the document. For segments that
           are still too large, it moves to the next separator in the list
           and repeats the process.

        Args:
            text: Current text segment.
            separators: Remaining separators to try.

        Returns:
            A list of chunks for the current text segment.
        """
        if not separators:
            return [text]
        final_chunks = []

        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]

        if separator != "":
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
        """Combines smaller splits into chunks as close to chunk_size as possible.

           Splitting alone can result in many tiny chunks. Merging is the
           corrective step that consolidates these small pieces back together,
           ensuring we maximize the information density in each chunk while
           staying within the requested size limit.

        Args:
            splits: List of text segments to potentially merge.
            separator: The separator used to originally split these segments.

        Returns:
            A list of merged chunks.
        """
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
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the orientation of two vectors in hyperspace,
    making it invariant to the magnitude of the vectors. In text retrieval,
    this means it measures the semantic 'closeness' regardless of the length
    of the text segments (assuming normalized or consistently scaled embeddings).

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
        """Evaluates different chunking strategies on a given text.

           This utility allows developers to empirically determine which
           strategy produces the most balanced chunks for their specific dataset.
           By comparing metrics like chunk count and average length, one can
           fine-tune the retrieval system for both performance and accuracy.

        Args:
            text: Sample text to be chunked.
            chunk_size: Target size for the chunks.

        Returns:
            A dictionary containing stats for each chunking strategy.
        """
        fixed_size_chunker = FixedSizeChunker(chunk_size=chunk_size)
        by_sentences_chunker = SentenceChunker()
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        res = {}
        for name, chunker in (
            ("fixed_size", fixed_size_chunker),
            ("by_sentences", by_sentences_chunker),
            ("recursive", recursive_chunker),
        ):
            chunks = chunker.chunk(text)
            res[name] = {}
            res[name]["chunks"] = chunks
            res[name]["count"] = len(chunks)
            res[name]["avg_length"] = sum((len(chunk) for chunk in chunks)) / len(
                chunks
            )

        return res
