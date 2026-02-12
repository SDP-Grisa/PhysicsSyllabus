"""
Embedding Generation and Vector Store Management
ChromaDB-only implementation (FAISS removed for Kaggle compatibility)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector Store
import chromadb
from chromadb.config import Settings

from schemas import ChunkMetadata
from config import (
    EMBEDDING_CONFIG,
    VECTOR_STORE_CONFIG,
    VECTOR_DB_DIR
)


# ============================================================
# Embedding Generator
# ============================================================

class EmbeddingGenerator:
    """Generates embeddings using open-source models"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.device = EMBEDDING_CONFIG["device"]

        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Embedding dimension: {self.dimension}")

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for list of texts"""
        batch_size = batch_size or EMBEDDING_CONFIG["batch_size"]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=EMBEDDING_CONFIG["normalize_embeddings"],
            convert_to_numpy=True
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.encode([text], show_progress=False)[0]


# ============================================================
# Base Vector Store
# ============================================================

class VectorStore:
    """Base class for vector stores"""

    def add(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


# ============================================================
# ChromaDB Store (ONLY STORE NOW)
# ============================================================

class ChromaDBStore(VectorStore):
    """ChromaDB vector store implementation"""

    def __init__(self, persist_directory: Optional[Path] = None):
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        self.collection_name = VECTOR_STORE_CONFIG["collection_name"]

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=VECTOR_STORE_CONFIG["anonymized_telemetry"]
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": VECTOR_STORE_CONFIG["distance_metric"]}
            )
            print(f"Created new collection: {self.collection_name}")

    # --------------------------------------------------------

    def add(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
        """Add chunks to ChromaDB"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text_content for chunk in chunks]
        metadatas = [chunk.get_search_metadata() for chunk in chunks]
        embeddings_list = embeddings.tolist()

        batch_size = 100

        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, len(chunks))

            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=embeddings_list[i:end_idx]
            )

        print(f"Added {len(chunks)} chunks to ChromaDB")

    # --------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """Search for similar chunks"""

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters if filters else None
        )

        formatted_results = []

        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                metadata["text_content"] = results["documents"][0][i]

                distance = results["distances"][0][i]
                score = 1 - distance  # cosine similarity

                formatted_results.append((metadata, score))

        return formatted_results

    # --------------------------------------------------------

    def save(self):
        """ChromaDB auto-persists"""
        print(f"ChromaDB persisted to: {self.persist_directory}")

    def load(self):
        """ChromaDB auto-loads"""
        pass

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": str(self.persist_directory)
        }


# ============================================================
# Factory Function
# ============================================================

def create_vector_store() -> VectorStore:
    """
    Create ChromaDB vector store
    (FAISS removed for simplicity & Kaggle compatibility)
    """
    return ChromaDBStore()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import json

    # Load chunks
    with open("data/metadata/chunks.json", "r") as f:
        chunks_data = json.load(f)

    chunks = [ChunkMetadata.from_dict(c) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks")

    # Generate embeddings
    embedder = EmbeddingGenerator()
    texts = [chunk.text_content for chunk in chunks]
    embeddings = embedder.encode(texts)

    print(f"Generated embeddings shape: {embeddings.shape}")

    # Create vector store
    vector_store = create_vector_store()
    vector_store.add(chunks, embeddings)
    vector_store.save()

    print("\nVector store created successfully!")

    # Test search
    query = "What is Young's modulus?"
    query_embedding = embedder.encode_single(query)
    results = vector_store.search(query_embedding, top_k=5)

    print(f"\nTest search for: '{query}'")
    for i, (metadata, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Chapter: {metadata['chapter_name']}")
        print(f"   Text: {metadata['text_content'][:100]}...")
