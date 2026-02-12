"""
Build Index Script
Processes NCERT Physics PDFs and builds the vector database

Usage:
    python build_index.py --pdf path/to/physics_class11.pdf --class 11
    python build_index.py --pdf path/to/physics_class12.pdf --class 12
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys

from pdf_extraction import PDFExtractor, save_extraction_results
from chunking import ChunkingStrategy, ChunkLinker
from embeddings import EmbeddingGenerator, create_vector_store
from schemas import ChunkMetadata, Chapter, Document
from config import (
    PDF_DIR,
    METADATA_DIR,
    CHUNK_CONFIG,
    EMBEDDING_CONFIG,
    VECTOR_STORE_CONFIG
)


class IndexBuilder:
    """
    Orchestrates the complete indexing pipeline
    """
    
    def __init__(self, class_level: str):
        self.class_level = class_level
        self.document = Document(
            book_title=f"NCERT Physics Class {class_level}",
            class_level=class_level
        )
        
        # Initialize components
        self.embedder = None
        self.vector_store = None
        self.all_chunks: List[ChunkMetadata] = []
        
    def build_from_pdf(self, pdf_path: str, chapter_info: Dict = None):
        """
        Build index from a single PDF
        
        Args:
            pdf_path: Path to PDF file
            chapter_info: Optional chapter metadata
        """
        print("="*80)
        print("NCERT PHYSICS RAG - INDEX BUILDING")
        print("="*80)
        
        # Step 1: Extract content from PDF
        print("\n[STEP 1/5] Extracting content from PDF...")
        print(f"PDF: {pdf_path}")
        
        with PDFExtractor(pdf_path) as extractor:
            extracted_content = extractor.extract_all()
        
        # Save extraction results
        extraction_file = METADATA_DIR / f"extraction_{self.class_level}.json"
        save_extraction_results(extracted_content, extraction_file)
        
        # Step 2: Chunk the content
        print("\n[STEP 2/5] Chunking content...")
        chunker = ChunkingStrategy(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            overlap=CHUNK_CONFIG["chunk_overlap"]
        )
        
        chunks = chunker.chunk_document(extracted_content, chapter_info)
        
        # Link related chunks
        print("Linking related chunks...")
        chunks = ChunkLinker.link_diagrams_to_text(chunks)
        chunks = ChunkLinker.link_formulas(chunks)
        
        self.all_chunks.extend(chunks)
        
        # Save chunks
        chunks_file = METADATA_DIR / f"chunks_{self.class_level}.json"
        chunks_data = [chunk.to_dict() for chunk in chunks]
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to: {chunks_file}")
        
        # Step 3: Generate embeddings
        print("\n[STEP 3/5] Generating embeddings...")
        if self.embedder is None:
            self.embedder = EmbeddingGenerator(
                model_name=EMBEDDING_CONFIG["model_name"]
            )
        
        texts = [chunk.text_content for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress=True)
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Step 4: Create/update vector store
        print("\n[STEP 4/5] Building vector store...")
        if self.vector_store is None:
            self.vector_store = create_vector_store(
                store_type=VECTOR_STORE_CONFIG["type"],
                dimension=self.embedder.dimension
            )
        
        self.vector_store.add(chunks, embeddings)
        
        # Step 5: Save vector store
        print("\n[STEP 5/5] Saving vector store...")
        self.vector_store.save()
        
        print("\n" + "="*80)
        print("INDEX BUILDING COMPLETE!")
        print("="*80)
        print(f"\nTotal chunks indexed: {len(self.all_chunks)}")
        print(f"Embedding model: {EMBEDDING_CONFIG['model_name']}")
        print(f"Vector store: {VECTOR_STORE_CONFIG['type']}")
        print(f"Storage location: {VECTOR_STORE_CONFIG['persist_directory']}")
        
    def build_from_multiple_pdfs(self, pdf_paths: List[str], chapters_info: List[Dict]):
        """
        Build index from multiple PDFs (e.g., separate PDFs for each chapter)
        
        Args:
            pdf_paths: List of PDF paths
            chapters_info: List of chapter metadata dicts
        """
        for pdf_path, chapter_info in zip(pdf_paths, chapters_info):
            print(f"\nProcessing: {pdf_path}")
            self.build_from_pdf(pdf_path, chapter_info)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the built index"""
        stats = {
            "total_chunks": len(self.all_chunks),
            "class_level": self.class_level,
            "chunks_by_type": {},
            "chunks_by_chapter": {},
            "total_diagrams": 0,
            "total_formulas": 0,
            "total_examples": 0
        }
        
        for chunk in self.all_chunks:
            # Count by type
            chunk_type = chunk.chunk_type
            stats["chunks_by_type"][chunk_type] = \
                stats["chunks_by_type"].get(chunk_type, 0) + 1
            
            # Count by chapter
            chapter = chunk.chapter_name
            stats["chunks_by_chapter"][chapter] = \
                stats["chunks_by_chapter"].get(chapter, 0) + 1
            
            # Count specific types
            if chunk_type == "diagram":
                stats["total_diagrams"] += 1
            elif chunk_type == "formula":
                stats["total_formulas"] += 1
            elif chunk_type == "example":
                stats["total_examples"] += 1
        
        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build RAG index for NCERT Physics textbooks"
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to PDF file"
    )
    
    parser.add_argument(
        "--class",
        dest="class_level",
        type=str,
        choices=["11", "12"],
        required=True,
        help="Class level (11 or 12)"
    )
    
    parser.add_argument(
        "--chapter-name",
        type=str,
        help="Chapter name (optional)"
    )
    
    parser.add_argument(
        "--chapter-number",
        type=int,
        help="Chapter number (optional)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics after building"
    )
    
    args = parser.parse_args()
    
    # Validate PDF exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Build chapter info if provided
    chapter_info = None
    if args.chapter_name or args.chapter_number:
        chapter_info = {}
        if args.chapter_name:
            chapter_info["chapter_name"] = args.chapter_name
        if args.chapter_number:
            chapter_info["chapter_number"] = args.chapter_number
    
    # Create index builder
    builder = IndexBuilder(class_level=args.class_level)
    
    # Build index
    try:
        builder.build_from_pdf(str(pdf_path), chapter_info)
        
        # Show stats if requested
        if args.stats:
            print("\n" + "="*80)
            print("INDEX STATISTICS")
            print("="*80)
            
            stats = builder.get_index_stats()
            
            print(f"\nTotal chunks: {stats['total_chunks']}")
            print(f"Class level: {stats['class_level']}")
            
            print("\nChunks by type:")
            for chunk_type, count in sorted(stats['chunks_by_type'].items()):
                print(f"  {chunk_type}: {count}")
            
            print("\nSpecial content:")
            print(f"  Diagrams: {stats['total_diagrams']}")
            print(f"  Formulas: {stats['total_formulas']}")
            print(f"  Examples: {stats['total_examples']}")
            
            if stats['chunks_by_chapter']:
                print("\nChunks by chapter:")
                for chapter, count in sorted(stats['chunks_by_chapter'].items()):
                    print(f"  {chapter}: {count}")
        
        print("\n✅ Index built successfully!")
        print("\nYou can now run the application:")
        print("  streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()