"""
Vector Database System with Multi-Modal Embeddings
Handles text, equations, and image embeddings
"""
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

class MultimodalVectorDB:
    """
    Vector database with support for:
    - Text embeddings (for paragraphs, concepts)
    - Equation embeddings
    - Image embeddings (optional with CLIP)
    - Metadata filtering (chapter, type, page)
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.text_collection     = self.client.get_or_create_collection("text_chunks",   metadata={"hnsw:space": "cosine"})
        self.equation_collection = self.client.get_or_create_collection("equations",     metadata={"hnsw:space": "cosine"})
        self.problem_collection  = self.client.get_or_create_collection("problems",      metadata={"hnsw:space": "cosine"})
        self.table_collection    = self.client.get_or_create_collection("tables",        metadata={"hnsw:space": "cosine"})
        self.image_collection    = self.client.get_or_create_collection("images",        metadata={"hnsw:space": "cosine"})

    def _create_metadata(self, chunk: Dict) -> Dict:
        """Create metadata for filtering"""
        metadata = {
            'page': chunk.get('page', 0),
            'type': chunk.get('type', 'text'),
        }
        
        if chunk.get('chapter'):
            metadata['chapter_number'] = chunk['chapter'].get('chapter_number', 0)
            metadata['chapter_title'] = chunk['chapter'].get('chapter_title', '')
        
        return metadata
    
    def add_text_chunks(self, chunks: List[Dict]):
        """Add text chunks to vector DB"""
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} text chunks to vector DB...")
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        ids = [f"text_{i}" for i in range(len(chunks))]
        metadatas = [self._create_metadata(chunk) for chunk in chunks]
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.text_collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
    
    def add_equations(self, equations: List[Dict]):
        """Add equations to vector DB"""
        if not equations:
            return
        
        print(f"Adding {len(equations)} equations to vector DB...")
        
        texts = [eq['content'] for eq in equations]
        embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        ids = [f"eq_{i}" for i in range(len(equations))]
        metadatas = [self._create_metadata(eq) for eq in equations]
        
        batch_size = 100
        for i in range(0, len(equations), batch_size):
            batch_end = min(i + batch_size, len(equations))
            
            self.equation_collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
    
    def add_problems(self, problems: List[Dict]):
        """Add problems/sums to vector DB"""
        if not problems:
            return
        
        print(f"Adding {len(problems)} problems to vector DB...")
        
        texts = [prob['content'] for prob in problems]
        embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        ids = [f"prob_{i}" for i in range(len(problems))]
        metadatas = [self._create_metadata(prob) for prob in problems]
        
        batch_size = 100
        for i in range(0, len(problems), batch_size):
            batch_end = min(i + batch_size, len(problems))
            
            self.problem_collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
    
    def add_tables(self, tables: List[Dict]):
        """Add tables to vector DB"""
        if not tables:
            return
        
        print(f"Adding {len(tables)} tables to vector DB...")
        
        # Use markdown representation for embedding
        texts = [table['markdown'] for table in tables]
        embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        ids = [f"table_{i}" for i in range(len(tables))]
        metadatas = [self._create_metadata(table) for table in tables]
        
        # Store full table data as document
        documents = [json.dumps(table) for table in tables]
        
        batch_size = 100
        for i in range(0, len(tables), batch_size):
            batch_end = min(i + batch_size, len(tables))
            
            self.table_collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
    
    def add_images(self, images: List[Dict]):
        """Add images to vector DB"""
        if not images:
            return
        
        print(f"Adding {len(images)} images to vector DB...")
        
        # Use OCR text for embedding
        texts = [img.get('ocr_text', '') or f"Image from page {img['page']}" for img in images]
        embeddings = self.text_model.encode(texts, show_progress_bar=True)
        
        ids = [f"img_{i}" for i in range(len(images))]
        metadatas = [self._create_metadata(img) for img in images]
        
        # Store image metadata as document
        documents = [json.dumps(img) for img in images]
        
        batch_size = 100
        for i in range(0, len(images), batch_size):
            batch_end = min(i + batch_size, len(images))
            
            self.image_collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
    
    def search_text(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict]:
        where = {'chapter_number': chapter_filter} if chapter_filter else None
        return self._search_with_rerank(self.text_collection, query, n_results, where=where)

    def search_equations(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict]:
        where = {'chapter_number': chapter_filter} if chapter_filter else None
        return self._search_with_rerank(self.equation_collection, query, n_results, where=where)

    def search_problems(self, query: str, n_results: int = 5, chapter_filter: Optional[int] = None) -> List[Dict]:
        where = {'chapter_number': chapter_filter} if chapter_filter else None
        return self._search_with_rerank(self.problem_collection, query, n_results, where=where)

    def search_tables(self, query: str, n_results: int = 3, chapter_filter: Optional[int] = None) -> List[Dict]:
        where = {'chapter_number': chapter_filter} if chapter_filter else None
        res = self._search_with_rerank(self.table_collection, query, n_results * 2, where=where)
        for item in res:
            item['table_data'] = json.loads(item['content'])
        return res[:n_results]

    def search_images(self, query: str, n_results: int = 3, chapter_filter: Optional[int] = None) -> List[Dict]:
        """Search for images - FIXED: Properly handle where clause"""
        # Start with chapter filter if provided
        where = None
        if chapter_filter:
            where = {'chapter_number': chapter_filter}
        
        # Boost for stress-strain related diagrams
        if 'stress' in query.lower() and 'strain' in query.lower():
            where = {'chapter_number': 9}  # Mechanical Properties of Solids is usually Ch.9
        
        print(f"🔍 Image search - query: '{query}', where: {where}")
        
        res = self._search_with_rerank(self.image_collection, query, n_results * 2, where=where)
        for item in res:
            item['image_data'] = json.loads(item['content'])
        return res[:n_results]
    
    def _search_with_rerank(self, collection, query: str, fetch_k: int, where: Optional[Dict] = None) -> List[Dict]:
        """Search with reranking - FIXED: Only pass where if it's not None"""
        print(f"  🔎 _search_with_rerank called with where={where}")
        
        q_emb = self.text_model.encode([query])[0]
        
        # Only include where parameter if it's actually set
        query_params = {
            'query_embeddings': [q_emb.tolist()],
            'n_results': fetch_k
        }
        
        if where is not None:
            query_params['where'] = where
        
        raw = collection.query(**query_params)
        candidates = self._format_results(raw)
        
        if not candidates:
            print(f"  ⚠️ No candidates found")
            return []
        
        print(f"  ✅ Found {len(candidates)} candidates, reranking...")
        
        pairs = [(query, c['content']) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked_idx = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_idx[:fetch_k//2]]

    def search_all(self, query: str, chapter_filter: Optional[int] = None) -> Dict:
        """Search across all content types"""
        return {
            'text': self.search_text(query, n_results=5, chapter_filter=chapter_filter),
            'equations': self.search_equations(query, n_results=3, chapter_filter=chapter_filter),
            'problems': self.search_problems(query, n_results=3, chapter_filter=chapter_filter),
            'tables': self.search_tables(query, n_results=2, chapter_filter=chapter_filter),
            'images': self.search_images(query, n_results=2, chapter_filter=chapter_filter)
        }
    
    def _format_results(self, results) -> List[Dict]:
        if not results['ids'] or not results['ids'][0]:
            return []
        return [{
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None
        } for i in range(len(results['ids'][0]))]

    def get_chapter_summary(self, chapter_number: int, max_chunks: int = 20) -> List[Dict]:
        res = self.text_collection.get(where={'chapter_number': chapter_number}, limit=max_chunks)
        return self._format_get_results(res)
    
    def _format_get_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB get results"""
        formatted = []
        
        if not results['ids']:
            return formatted
        
        for i in range(len(results['ids'])):
            formatted.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return formatted


def build_vector_db(content_file: str = "extracted_content/extracted_content.json"):
    """Build vector database from extracted content"""
    print("Loading extracted content...")
    with open(content_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    print("Initializing vector database...")
    db = MultimodalVectorDB()
    
    # Add all content
    db.add_text_chunks(content['text_chunks'])
    db.add_equations(content['equations'])
    db.add_problems(content['problems'])
    db.add_tables(content['tables'])
    db.add_images(content['images'])
    
    print("\nVector database built successfully!")
    return db


if __name__ == "__main__":
    # Build database from extracted content
    db = build_vector_db()