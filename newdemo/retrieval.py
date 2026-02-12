"""
Multi-Stage Retrieval Pipeline with Hybrid Search and Reranking
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from schemas import ChunkMetadata, RetrievalResult
from embeddings import EmbeddingGenerator, VectorStore
from config import RETRIEVAL_CONFIG, RERANKER_CONFIG, QUERY_INTENTS


@dataclass
class QueryIntent:
    """Detected intent for a user query"""
    primary_intent: str
    needs_diagram: bool
    needs_formula: bool
    chapter_filter: Optional[int] = None
    topic_filter: Optional[str] = None


class QueryAnalyzer:
    """
    Analyzes user queries to understand intent and requirements
    """
    
    @staticmethod
    def analyze(query: str) -> QueryIntent:
        """
        Analyze query to detect intent
        
        Returns:
            QueryIntent object with detected requirements
        """
        query_lower = query.lower()
        
        # Detect primary intent
        primary_intent = "concept_explanation"  # default
        
        for intent_type, keywords in QUERY_INTENTS.items():
            if any(keyword in query_lower for keyword in keywords):
                primary_intent = intent_type
                break
        
        # Detect specific needs
        needs_diagram = any(
            keyword in query_lower 
            for keyword in ["diagram", "figure", "graph", "illustration", "show"]
        )
        
        needs_formula = any(
            keyword in query_lower
            for keyword in ["formula", "equation", "derive", "expression"]
        )
        
        # Detect chapter filter
        chapter_match = re.search(r'chapter\s*(\d+)', query_lower)
        chapter_filter = int(chapter_match.group(1)) if chapter_match else None
        
        # Detect topic mentions (simple extraction)
        topic_filter = QueryAnalyzer._extract_topic(query)
        
        return QueryIntent(
            primary_intent=primary_intent,
            needs_diagram=needs_diagram,
            needs_formula=needs_formula,
            chapter_filter=chapter_filter,
            topic_filter=topic_filter
        )
    
    @staticmethod
    def _extract_topic(query: str) -> Optional[str]:
        """
        Extract topic/concept from query
        Simple heuristic: capitalize consecutive words
        """
        # Look for capitalized terms or quoted phrases
        matches = re.findall(r'"([^"]+)"', query)
        if matches:
            return matches[0]
        
        # Look for capitalized consecutive words
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        if matches:
            return matches[0]
        
        return None


class HybridRetriever:
    """
    Implements hybrid retrieval combining:
    - Dense vector search
    - Sparse BM25 search
    - Metadata filtering
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator,
        chunks_metadata: List[Dict]
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks_metadata = chunks_metadata
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25_index = self._build_bm25_index()
        print("BM25 index ready")
    
    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index for keyword search"""
        # Tokenize all documents
        tokenized_corpus = []
        for chunk in self.chunks_metadata:
            text = chunk.get('text_content', '')
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)
        
        return BM25Okapi(tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def retrieve(
        self,
        query: str,
        intent: QueryIntent,
        top_k: Optional[int] = None
    ) -> List[Tuple[Dict, float, str]]:
        """
        Hybrid retrieval combining vector and BM25
        
        Args:
            query: User query
            intent: Detected query intent
            top_k: Number of results to return
        
        Returns:
            List of (metadata, score, method) tuples
        """
        top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        
        # Build metadata filters
        filters = self._build_filters(intent)
        
        # 1. Vector search
        vector_results = self._vector_search(query, top_k * 2, filters)
        
        # 2. BM25 search
        bm25_results = self._bm25_search(query, top_k * 2, filters)
        
        # 3. Merge results
        if RETRIEVAL_CONFIG["use_hybrid"]:
            merged_results = self._merge_results(
                vector_results,
                bm25_results,
                vector_weight=RETRIEVAL_CONFIG["vector_weight"],
                bm25_weight=RETRIEVAL_CONFIG["bm25_weight"]
            )
        else:
            merged_results = vector_results
        
        # 4. Filter by similarity threshold
        filtered_results = [
            (metadata, score, method)
            for metadata, score, method in merged_results
            if score >= RETRIEVAL_CONFIG["similarity_threshold"]
        ]
        
        return filtered_results[:top_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> List[Tuple[Dict, float, str]]:
        """Dense vector search"""
        # Generate query embedding
        query_embedding = self.embedder.encode_single(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        return [(metadata, score, "vector") for metadata, score in results]
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> List[Tuple[Dict, float, str]]:
        """Sparse BM25 search"""
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        # Filter and format results
        results = []
        for idx in top_indices:
            if idx >= len(self.chunks_metadata):
                continue
            
            metadata = self.chunks_metadata[idx]
            score = scores[idx]
            
            # Apply filters
            if filters:
                if not self._matches_filters(metadata, filters):
                    continue
            
            # Normalize BM25 score to [0, 1]
            normalized_score = min(score / 10.0, 1.0)
            
            results.append((metadata, normalized_score, "bm25"))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _merge_results(
        self,
        vector_results: List[Tuple[Dict, float, str]],
        bm25_results: List[Tuple[Dict, float, str]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Tuple[Dict, float, str]]:
        """
        Merge vector and BM25 results using weighted scores
        """
        # Create score dict by chunk_id
        merged_scores = {}
        
        # Add vector results
        for metadata, score, _ in vector_results:
            chunk_id = metadata.get('chunk_id')
            merged_scores[chunk_id] = {
                'metadata': metadata,
                'vector_score': score,
                'bm25_score': 0.0
            }
        
        # Add/update with BM25 results
        for metadata, score, _ in bm25_results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id in merged_scores:
                merged_scores[chunk_id]['bm25_score'] = score
            else:
                merged_scores[chunk_id] = {
                    'metadata': metadata,
                    'vector_score': 0.0,
                    'bm25_score': score
                }
        
        # Calculate combined scores
        results = []
        for chunk_id, data in merged_scores.items():
            combined_score = (
                vector_weight * data['vector_score'] +
                bm25_weight * data['bm25_score']
            )
            results.append((data['metadata'], combined_score, "hybrid"))
        
        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _build_filters(self, intent: QueryIntent) -> Optional[Dict]:
        """Build metadata filters from query intent"""
        filters = {}
        
        if intent.chapter_filter:
            filters['chapter_number'] = intent.chapter_filter
        
        # Note: ChromaDB supports simple equality filters
        # For more complex filtering, do it post-retrieval
        
        return filters if filters else None
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True


class Reranker:
    """
    Reranks retrieved results using cross-encoder
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or RERANKER_CONFIG["model_name"]
        self.device = RERANKER_CONFIG["device"]
        
        print(f"Loading reranker: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=512
        )
        print("Reranker loaded")
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[Dict, float, str]],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Original query
            results: Retrieved results (metadata, score, method)
            top_k: Number of top results to return
        
        Returns:
            List of RetrievalResult objects, sorted by rerank score
        """
        if not results:
            return []
        
        top_k = top_k or RETRIEVAL_CONFIG["top_k_rerank"]
        
        # Prepare pairs for reranking
        pairs = []
        metadatas = []
        
        for metadata, score, method in results:
            text = metadata.get('text_content', '')
            pairs.append([query, text])
            metadatas.append((metadata, score, method))
        
        # Get rerank scores
        rerank_scores = self.model.predict(
            pairs,
            batch_size=RERANKER_CONFIG["batch_size"],
            show_progress_bar=False
        )
        
        # Create RetrievalResult objects
        reranked_results = []
        
        for i, score in enumerate(rerank_scores):
            metadata, original_score, method = metadatas[i]
            
            # Convert metadata dict to ChunkMetadata
            chunk = ChunkMetadata(
                chunk_id=metadata.get('chunk_id', ''),
                chunk_type=metadata.get('chunk_type', 'text'),
                chapter_name=metadata.get('chapter_name', ''),
                chapter_number=metadata.get('chapter_number', 0),
                page_number=metadata.get('page_number', 0),
                text_content=metadata.get('text_content', ''),
                topic=metadata.get('topic'),
                subtopic=metadata.get('subtopic'),
                figure_id=metadata.get('figure_id'),
                image_path=metadata.get('image_path'),
                caption=metadata.get('caption')
            )
            
            result = RetrievalResult(
                chunk=chunk,
                score=float(score),
                rank=i,
                retrieval_method="reranked"
            )
            
            reranked_results.append(result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
        
        return reranked_results[:top_k]


class RetrievalPipeline:
    """
    Complete retrieval pipeline with all stages
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator,
        chunks_metadata: List[Dict]
    ):
        self.query_analyzer = QueryAnalyzer()
        self.retriever = HybridRetriever(vector_store, embedder, chunks_metadata)
        
        if RETRIEVAL_CONFIG["enable_reranking"]:
            self.reranker = Reranker()
        else:
            self.reranker = None
    
    def retrieve(
        self,
        query: str,
        chapter_filter: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], QueryIntent]:
        """
        Full retrieval pipeline
        
        Args:
            query: User query
            chapter_filter: Optional chapter number filter
        
        Returns:
            (retrieval_results, query_intent)
        """
        # 1. Analyze query
        intent = self.query_analyzer.analyze(query)
        
        # Apply manual chapter filter if provided
        if chapter_filter:
            intent.chapter_filter = chapter_filter
        
        # 2. Retrieve
        initial_results = self.retriever.retrieve(query, intent)
        
        # 3. Rerank if enabled
        if self.reranker:
            final_results = self.reranker.rerank(query, initial_results)
        else:
            # Convert to RetrievalResult
            final_results = []
            for i, (metadata, score, method) in enumerate(initial_results):
                chunk = ChunkMetadata(
                    chunk_id=metadata.get('chunk_id', ''),
                    chunk_type=metadata.get('chunk_type', 'text'),
                    chapter_name=metadata.get('chapter_name', ''),
                    chapter_number=metadata.get('chapter_number', 0),
                    page_number=metadata.get('page_number', 0),
                    text_content=metadata.get('text_content', ''),
                    topic=metadata.get('topic'),
                    image_path=metadata.get('image_path')
                )
                
                result = RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=i + 1,
                    retrieval_method=method
                )
                final_results.append(result)
        
        return final_results, intent


if __name__ == "__main__":
    # Example usage will be in build_index.py
    pass