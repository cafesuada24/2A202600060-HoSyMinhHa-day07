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
        """Initializes the KnowledgeBaseAgent.

           The agent acts as an orchestrator in a Retrieval-Augmented Generation (RAG)
           pipeline. It decouples the storage mechanism (EmbeddingStore) from the
           generative model (llm_fn), allowing for flexible experimentation with
           different vector databases or LLM providers.

        Args:
            store: An instance of EmbeddingStore used to retrieve relevant context.
            llm_fn: A callable that takes a string prompt and returns a string response,
                representing the interface to a Large Language Model.
        """
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """Generates an answer to the given question using retrieved context.

           This method implements the core RAG logic. It first performs a semantic
           search in the vector store to find the most relevant 'top_k' information
           snippets. These snippets are aggregated into a single context string.
           The final prompt is constructed by injecting this context into a
           predefined template, which instructs the LLM to answer the question
           *only* based on the provided data, thereby reducing hallucinations.

        Args:
            question: The user's query or question.
            top_k: The number of most relevant chunks to retrieve from the store.
                Defaults to 3.

        Returns:
            The generated answer from the LLM.
        """
        context = "\n---\n".join(
            x["content"] for x in self._store.search(query=question, top_k=top_k)
        )
        augmented_input = self.TEMPLATE.format(context=context)
        return self._llm_fn(augmented_input)
