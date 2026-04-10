from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    TEMPLATE = r"""<instruction>
    Answer the following question using the provied content.
    </instruction>

    <context>
    {context}
    </context>

    <question>
    </question>"""

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        context = "\n---\n".join(
            x["content"] for x in self._store.search(query=question, top_k=top_k)
        )
        augmented_input = self.TEMPLATE.format(context=context)
        return self._llm_fn(augmented_input)
