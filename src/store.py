from __future__ import annotations

from typing import Any, Callable
import heapq

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": None,
        }

    def _get_or_create_embedding(self, record: dict[str, Any]) -> list[float]:
        if record.get("embedding") is None:
            record["embedding"] = self._embedding_fn(record["content"])
        return record["embedding"]

    def _search_records(
        self,
        query: str,
        records: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)

        # nlargest is significantly faster when top_k << len(records)
        if not records:
            return []

        query_emb = self._embedding_fn(query)

        # Use nlargest for O(N log K) efficiency
        # We calculate the dot product as the sorting key
        top_items = heapq.nlargest(
            top_k,
            records,
            key=lambda x: _dot(self._get_or_create_embedding(x), query_emb),
        )

        # Transform to final format: only content and score
        return [
            {
                "content": item["content"],
                "score": _dot(item["embedding"], query_emb),
                "metadata": item["metadata"],
            }
            for item in top_items
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not self._use_chroma:
            self._store.extend([self._make_record(doc) for doc in docs])
            return

        if self._collection is None:
            raise RuntimeError("ChromaDB is not initialized")
        ids = [doc.id for doc in docs]
        documents = [doc.content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            return self.search_with_filter(query, top_k=top_k)

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if not self._use_chroma:
            return len(self._store)

        assert self._collection is not None
        return self._collection.count()

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None,
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            assert self._collection is not None
            # Chroma uses the 'where' parameter for metadata filtering
            query_result = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter,
            )

            # Reconstruct list of dicts to match internal format
            results = []
            # Chroma returns results as lists of lists (one per query text)
            for i in range(len(query_result["ids"][0])):
                results.append(
                    {
                        "content": query_result["documents"][0][i],
                        "score": query_result["distances"][0][i],
                        "metadata": query_result["metadatas"][0][i],
                    }
                )
            return results

        # --- In-Memory Implementation ---
        # 1. Pre-filter records based on metadata_filter
        filtered_records = self._store
        if metadata_filter:
            filtered_records = [
                rec
                for rec in self._store
                if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]

        # 2. Perform similarity search on the subset
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if not self._use_chroma:
            original_len = len(self._store)
            self._store = [doc for doc in self._store if doc["id"] != doc_id]
            return len(self._store) < original_len

        assert self._collection is not None
        res = self._collection.delete(ids=[doc_id])
        return res["deleted"] == 1
