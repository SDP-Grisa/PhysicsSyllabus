"""
Configuration file for NCERT Physics RAG System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGES_DIR = DATA_DIR / "extracted_images"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
METADATA_DIR = DATA_DIR / "metadata"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PDF_DIR, IMAGES_DIR, VECTOR_DB_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# PDF Processing Configuration
PDF_CONFIG = {
    "dpi": 300,
    "extract_images": True,
    "extract_tables": True,
    "min_figure_height": 50,  # minimum height in pixels
    "min_figure_width": 50,   # minimum width in pixels
}

# Chunking Configuration
CHUNK_CONFIG = {
    "chunk_size": 500,  # tokens
    "chunk_overlap": 75,  # tokens
    "min_chunk_size": 100,  # minimum tokens per chunk
    "preserve_boundaries": [
        "Example",
        "Definition",
        "Formula",
        "Summary",
        "Exercise",
        "Question"
    ]
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-large-en-v1.5",  # Best open-source embedding model
    "dimension": 1024,
    "device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu",
    "batch_size": 32,
    "normalize_embeddings": True
}

# Alternative embedding models (ranked by performance)
EMBEDDING_ALTERNATIVES = [
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-v2",
    "nomic-ai/nomic-embed-text-v1"
]

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "type": "chromadb",  # or "faiss"
    "collection_name": "ncert_physics",
    "distance_metric": "cosine",
    "persist_directory": str(VECTOR_DB_DIR),
    "anonymized_telemetry": False
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "top_k": 10,  # initial retrieval
    "top_k_rerank": 5,  # after reranking
    "similarity_threshold": 0.3,
    "use_hybrid": True,  # vector + BM25
    "bm25_weight": 0.3,
    "vector_weight": 0.7,
    "enable_reranking": True
}

# Reranking Configuration
RERANKER_CONFIG = {
    "model_name": "BAAI/bge-reranker-large",
    "device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu",
    "batch_size": 16
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",  # Default model
    "device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu",
    "load_in_8bit": True,  # Memory optimization
    "max_new_tokens": 1024,
    "temperature": 0.1,  # Low temperature for factual answers
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}

# Alternative LLM models (ranked by recommendation)
LLM_ALTERNATIVES = [
    "meta-llama/Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-medium-4k-instruct",
    "google/gemma-7b-it"
]

# Metadata Schema
METADATA_SCHEMA = {
    "required_fields": [
        "chunk_id",
        "chunk_type",
        "chapter_name",
        "chapter_number",
        "page_number",
        "text_content"
    ],
    "optional_fields": [
        "topic",
        "subtopic",
        "figure_id",
        "bbox",
        "caption",
        "related_formula_ids",
        "difficulty_level",
        "image_path"
    ],
    "chunk_types": [
        "text",
        "formula",
        "diagram",
        "table",
        "example",
        "definition",
        "summary",
        "exercise"
    ]
}

# Prompt Templates
SYSTEM_PROMPT = """You are an expert NCERT Physics tutor for Class 11 and 12 students. 
Your role is to provide accurate, textbook-grounded answers based ONLY on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, clearly state "This information is not available in the provided context"
3. Cite the chapter and page number for all information
4. Preserve mathematical notation exactly as written
5. When diagrams are mentioned, reference them clearly
6. Never hallucinate or add information not in the context
7. Keep explanations clear and student-friendly
8. Use proper physics terminology from NCERT textbooks
"""

ANSWER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based strictly on the context above
- Cite sources: [Chapter {chapter_number}, Page {page_number}]
- If formulas are involved, write them clearly
- If diagrams are referenced, mention them explicitly
- If the answer is not in the context, say so clearly

Answer:"""

# Query Intent Classification
QUERY_INTENTS = {
    "concept_explanation": [
        "explain", "what is", "describe", "how does", "why"
    ],
    "formula_request": [
        "formula", "equation", "derive", "expression"
    ],
    "diagram_request": [
        "diagram", "figure", "graph", "illustration", "show"
    ],
    "example_request": [
        "example", "solve", "problem", "numerical"
    ],
    "definition": [
        "define", "definition", "meaning", "what is"
    ],
    "comparison": [
        "difference between", "compare", "versus", "vs"
    ],
    "summary": [
        "summary", "summarize", "overview", "brief"
    ]
}

# Streamlit UI Configuration
UI_CONFIG = {
    "page_title": "NCERT Physics RAG System",
    "page_icon": "📚",
    "layout": "wide",
    "sidebar_state": "expanded",
    "show_debug": True,
    "max_image_width": 600
}