# SABERT: Baseball Analytics RAG System ⚾

SABERT is a domain-specific Retrieval-Augmented Generation (RAG) system for baseball analytics, combining vector-based document retrieval, memory-augmented reasoning, and chain-of-thought refinement.

## Overview

The system retrieves information from a curated baseball analytics corpus using BERT embeddings and ChromaDB, and generates answers using Groq-hosted LLaMA models. It supports multiple reasoning strategies and persists conversational memory across queries.

## Key Features

- Vector-based document retrieval (ChromaDB + SentenceTransformers)
- Memory-augmented retrieval across past Q&A turns
- Multiple reasoning modes:
  - **Direct RAG**
  - **Chain-of-Thought refinement (self-verification)**
- Interactive Streamlit chat interface
- Command-line interface (CLI)
- Evaluation script for systematic comparison of reasoning modes

## Steps

1. Get Groq API Key
2. Build corpus (corpus_builder.py)
3. Run the Streamlit GUI (streamlit_gui.py) or CLI (main_cli.py)

