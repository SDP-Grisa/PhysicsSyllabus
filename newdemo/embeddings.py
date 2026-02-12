"""
Embedding Generation and Vector Store Management
Supports ChromaDB and FAISS with multiple embedding models
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector Stores
import chromadb
from chromadb.config import Settings
import faiss

from schemas import ChunkMetadata, MetadataValidator
from config import (
    EMBEDDING_CONFIG,
    VECTOR_STORE_CONFIG,
    VECTOR_DB_DIR
)


class EmbeddingGenerator:
    """
    Generates embeddings using open-source models
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.device = EMBEDDING_CONFIG["device"]
        
        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
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


class VectorStore:
    """
    Base class for vector stores
    """
    
    def add(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
        """Add chunks with embeddings"""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[ChunkMetadata, float]]:
        """Search for similar chunks"""
        raise NotImplementedError
    
    def save(self):
        """Persist vector store"""
        raise NotImplementedError
    
    def load(self):
        """Load vector store"""
        raise NotImplementedError


class ChromaDBStore(VectorStore):
    """
    ChromaDB vector store implementation
    """
    
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
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": VECTOR_STORE_CONFIG["distance_metric"]}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
        """Add chunks to ChromaDB"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text_content for chunk in chunks]
        metadatas = [chunk.get_search_metadata() for chunk in chunks]
        embeddings_list = embeddings.tolist()
        
        # Add to collection in batches
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
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"chapter_number": 9})
        
        Returns:
            List of (metadata_dict, score) tuples
        """
        # Build where filter if provided
        where_filter = None
        if filters:
            where_filter = filters
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        
        if results and results['ids']:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                metadata['text_content'] = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1 - distance
                
                formatted_results.append((metadata, score))
        
        return formatted_results
    
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


class FAISSStore(VectorStore):
    """
    FAISS vector store implementation (for better performance)
    """
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "IndexFlatIP":
            # Inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            # IVF for faster search on large datasets
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Storage for metadata
        self.id_to_metadata: Dict[int, Dict] = {}
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.current_idx = 0
    
    def add(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
        """Add chunks to FAISS"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Train index if needed
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings)
        
        # Add embeddings
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for chunk in chunks:
            metadata = chunk.get_search_metadata()
            metadata['text_content'] = chunk.text_content
            
            self.id_to_metadata[self.current_idx] = metadata
            self.chunk_id_to_idx[chunk.chunk_id] = self.current_idx
            self.current_idx += 1
        
        print(f"Added {len(chunks)} chunks to FAISS")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """Search for similar chunks"""
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k * 2  # Get more results for filtering
        )
        
        # Format results with filtering
        formatted_results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= self.current_idx:
                continue
            
            metadata = self.id_to_metadata[idx]
            
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            formatted_results.append((metadata, float(score)))
            
            if len(formatted_results) >= top_k:
                break
        
        return formatted_results
    
    def save(self, path: Optional[Path] = None):
        """Save FAISS index and metadata"""
        save_dir = path or VECTOR_DB_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = save_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = save_dir / "faiss_metadata.json"
        metadata = {
            "id_to_metadata": self.id_to_metadata,
            "chunk_id_to_idx": self.chunk_id_to_idx,
            "current_idx": self.current_idx,
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"FAISS index saved to: {save_dir}")
    
    def load(self, path: Optional[Path] = None):
        """Load FAISS index and metadata"""
        load_dir = path or VECTOR_DB_DIR
        
        # Load index
        index_path = load_dir / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = load_dir / "faiss_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Convert string keys back to integers for id_to_metadata
        self.id_to_metadata = {
            int(k): v for k, v in metadata["id_to_metadata"].items()
        }
        self.chunk_id_to_idx = metadata["chunk_id_to_idx"]
        self.current_idx = metadata["current_idx"]
        
        print(f"FAISS index loaded from: {load_dir}")


def create_vector_store(store_type: str = "chromadb", dimension: int = 1024) -> VectorStore:
    """
    Factory function to create vector store
    
    Args:
        store_type: "chromadb" or "faiss"
        dimension: Embedding dimension (for FAISS)
    
    Returns:
        VectorStore instance
    """
    if store_type == "chromadb":
        return ChromaDBStore()
    elif store_type == "faiss":
        return FAISSStore(dimension=dimension)
    else:
        raise ValueError(f"Unknown store type: {store_type}")


if __name__ == "__main__":
    # Example usage
    
    # Load chunks
    with open("data/metadata/chunks.json", 'r') as f:
        chunks_data = json.load(f)
    
    chunks = [ChunkMetadata.from_dict(c) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    texts = [chunk.text_content for chunk in chunks]
    embeddings = embedder.encode(texts)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create and populate vector store
    vector_store = create_vector_store(
        store_type=VECTOR_STORE_CONFIG["type"],
        dimension=embedder.dimension
    )
    
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