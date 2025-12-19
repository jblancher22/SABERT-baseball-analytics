# rag_core.py

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False




class BaseballRAGSystem:
    """
    BaseballRAGSystem - Focuses only on retrieval and reasoning.
    It assumes the ChromaDB vector store has already been built by corpus_builder.py.
    """

    def __init__(
            self,
            groq_api_key: Optional[str] = None,
            #embed_model_name: str="BAAI/bge-small-en-v1.5",
            embed_model_name: str='sentence-transformers/bert-base-nli-mean-tokens',
            chroma_dir: str = ".chroma_baseball",
    ):
        """
        Initialize the system.
        """

        # --- Embedding model (shared for corpus + memory) ---
        # This will still take time/memory to load, but it's essential for runtime similarity
        self.embed_model = SentenceTransformer(embed_model_name)

        # --- Resolve Groq API key securely ---
        if groq_api_key is None:
            # Priority:
            # 1. Streamlit secrets
            # 2. Environment variable
            if STREAMLIT_AVAILABLE and "GROQ_API_KEY" in st.secrets:
                groq_api_key = st.secrets["GROQ_API_KEY"]
            else:
                groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY in environment variables or .streamlit/secrets.toml"
            )

        self.client = Groq(api_key=groq_api_key)

        # --- Vector database client (ChromaDB) ---
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Collections (assumed to be pre-populated or will populate memory on first run)
        self.corpus_collection = self.chroma_client.get_or_create_collection(
            "baseball_corpus"
        )
        self.memory_collection = self.chroma_client.get_or_create_collection(
            "baseball_memory"
        )

        # Check for corpus existence and warn if empty
        if self.corpus_collection.count() == 0:
            print("⚠️ WARNING: Corpus is empty. Run corpus_builder.py first!")

        # Counter to generate unique memory IDs.
        self.turn_id_counter = self.memory_collection.count()
        print(f"RAG system initialized. Current memory turns: {self.turn_id_counter}")

    # -------------------------------------------------------------------------
    # The following methods are the same as in your original file:
    # -------------------------------------------------------------------------

    # BASIC RETRIEVAL
    def retrieve_corpus(
            self, query: str, k_docs: int = 10, max_context_chars: int = 8000
    ) -> Dict[str, Any]:
        """Retrieve top-k documents from the corpus."""

        q_emb = self.embed_model.encode(query).tolist()
        results = self.corpus_collection.query(
            query_embeddings=[q_emb],
            n_results=k_docs
        )

        if not results.get("documents", [[]])[0]:
            # print("\n[CRITICAL DEBUG] Corpus Retrieval FAILED: Zero documents returned.")
            return {"context": "", "sources": [], "raw": results}

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        fused = ""
        included = []

        for d, m in zip(docs, metas):
            candidate = f"\n\n[SOURCE: {m['source']}]\n{d}"
            if len(fused) + len(candidate) > max_context_chars:
                break
            fused += candidate
            included.append(m["source"])

        return {
            "context": fused,
            "sources": included,
            "raw": results,
        }

    # MEMORY-AUGMENTED RETRIEVAL
    def add_to_memory(self, query: str, answer: str) -> None:
        """Store a Q&A pair as an "episodic memory" in the memory collection."""

        combined = f"Q: {query}\nA: {answer}"
        emb = self.embed_model.encode(combined).tolist()
        mem_id = f"turn_{self.turn_id_counter}"
        self.turn_id_counter += 1

        self.memory_collection.add(
            ids=[mem_id],
            documents=[combined],
            metadatas=[{"type": "qa_turn"}],
            embeddings=[emb],
        )

    def retrieve_memory(self, query: str, k: int = 3) -> List[str]:
        """Retrieve up to k most similar past Q&A turns from memory."""
        if self.memory_collection.count() == 0:
            return []

        q_emb = self.embed_model.encode(query).tolist()
        results = self.memory_collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )
        return results["documents"][0]


    # LLM HELPER
    def call_llm(self, messages: List[Dict[str, str]], model: str = "llama-3.3-70b-versatile") -> str:
        """Convenience wrapper around Groq chat completions."""
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    # BASELINE DIRECT RAG ANSWER
    def format_memory(self, memory_snippets: List[str]) -> str:
        """Format retrieved memory Q&A pairs into explicit few-shot examples."""
        if not memory_snippets:
            return ""

        lines = ["FEW-SHOT EXAMPLES (similar past Q&A):"]
        for i, qa in enumerate(memory_snippets, start=1):
            lines.append(f"\nExample {i}:\n{qa}")
        return "\n".join(lines) + "\n\n"

    def answer_direct(self, query: str) -> Dict[str, Any]:
        """Simplest RAG mode: retrieve and answer."""

        corpus = self.retrieve_corpus(query)
        memory_snippets = self.retrieve_memory(query)
        examples_block = self.format_memory(memory_snippets)
        context = f"{examples_block}CORPUS CONTEXT:\n{corpus['context']}"

        if memory_snippets:
            context = (
                    "RECENT CONVERSATION SNIPPETS (may be helpful):\n"
                    + "\n\n".join(memory_snippets)
                    + "\n\nCORPUS CONTEXT:\n"
                    + context
            )

        system_prompt = (
            (
                "You are a baseball analytics assistant.\n"
                "You will see (1) a small set of example Q&A pairs that show how "
                "to answer similar questions, and (2) corpus context from baseball "
                "analytics documents.\n\n"
                "Treat the examples as in-context learning: imitate their style and "
                "logic, but base your answer only on information supported by the "
                "corpus context. If the answer is not supported by the context, say "
                "you don't know."
            )
        )

        answer = self.call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )
        self.add_to_memory(query, answer)

        return {
            "answer": answer,
            "sources": corpus["sources"],
            "mode": "direct_rag",
        }

    # CHAIN-OF-THOUGHT REFINEMENT (self-check / verification)
    def cot_refine(self, query: str) -> Dict[str, Any]:
        """Two-stage Chain-of-Thought refinement."""

        corpus = self.retrieve_corpus(query)
        memory_snippets = self.retrieve_memory(query)

        context = corpus["context"]
        if memory_snippets:
            context = (
                    "RECENT CONVERSATION SNIPPETS (may be helpful):\n"
                    + "\n\n".join(memory_snippets)
                    + "\n\nCORPUS CONTEXT:\n"
                    + context
            )

        # Step 1: Draft answer with hidden reasoning
        draft = self.call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a baseball analytics expert.\n"
                        "Given the context and question, first reason step by step in "
                        "a hidden chain-of-thought... After you finish reasoning, output a "
                        "section labeled 'DRAFT ANSWER:' followed by only a concise answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ]
        )

        # Step 2: Second pass to verify the draft
        refined = self.call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are checking another analyst's work. Compare the draft "
                        "against the context, correct any factual or logical errors, "
                        "and then output only a clean, final answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Draft answer from another analyst:\n{draft}\n\n"
                        f"Question: {query}\n\n"
                        "Verify and correct the draft if needed. Answer succinctly."
                    ),
                },
            ]
        )

        self.add_to_memory(query, refined)

        return {
            "answer": refined,
            "sources": corpus["sources"],
            "mode": "cot_refine",
        }
