"""
Semantic + Structural Chunking for NCERT Physics Content
Preserves topic boundaries, formulas, examples
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tiktoken

from schemas import ChunkMetadata, BoundingBox
from pdf_extraction import TextAnalyzer
from config import CHUNK_CONFIG


class ChunkingStrategy:
    """
    Implements semantic and structural chunking
    Respects topic boundaries and preserves context
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 75):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def chunk_document(
        self,
        extracted_content: Dict,
        chapter_info: Optional[Dict] = None
    ) -> List[ChunkMetadata]:
        """
        Main chunking pipeline
        
        Args:
            extracted_content: Output from PDFExtractor
            chapter_info: Optional chapter metadata
        
        Returns:
            List of ChunkMetadata objects
        """
        all_chunks = []
        
        # Process text chunks
        text_chunks = self._chunk_text_content(
            extracted_content["text"],
            chapter_info
        )
        all_chunks.extend(text_chunks)
        
        # Process figures as separate chunks
        figure_chunks = self._create_figure_chunks(
            extracted_content["figures"],
            chapter_info
        )
        all_chunks.extend(figure_chunks)
        
        # Process tables
        table_chunks = self._create_table_chunks(
            extracted_content.get("tables", {}),
            chapter_info
        )
        all_chunks.extend(table_chunks)
        
        print(f"\nChunking summary:")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Text chunks: {len(text_chunks)}")
        print(f"  Figure chunks: {len(figure_chunks)}")
        print(f"  Table chunks: {len(table_chunks)}")
        
        return all_chunks
    
    def _chunk_text_content(
        self,
        page_texts: Dict[int, str],
        chapter_info: Optional[Dict]
    ) -> List[ChunkMetadata]:
        """
        Chunk text content with semantic boundaries
        """
        chunks = []
        
        for page_num, text in page_texts.items():
            if not text or len(text.strip()) < 50:
                continue
            
            # Detect structural elements
            elements = self._detect_structural_elements(text)
            
            # Create chunks respecting boundaries
            page_chunks = self._create_chunks_from_elements(
                elements,
                page_num,
                chapter_info
            )
            
            chunks.extend(page_chunks)
        
        return chunks
    
    def _detect_structural_elements(self, text: str) -> List[Dict]:
        """
        Detect examples, definitions, formulas, etc.
        Returns list of structural elements with their positions
        """
        elements = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        current_pos = 0
        
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2
                continue
            
            # Detect element type
            element_type = TextAnalyzer.identify_chunk_type(para)
            
            # Detect if it's a heading
            is_heading = bool(re.match(r'^\d+\.?\d*\s+[A-Z]', para.strip()))
            
            element = {
                'text': para,
                'type': 'heading' if is_heading else element_type,
                'start_pos': current_pos,
                'end_pos': current_pos + len(para),
                'is_boundary': is_heading or element_type in ['example', 'definition']
            }
            
            elements.append(element)
            current_pos += len(para) + 2
        
        return elements
    
    def _create_chunks_from_elements(
        self,
        elements: List[Dict],
        page_num: int,
        chapter_info: Optional[Dict]
    ) -> List[ChunkMetadata]:
        """
        Create chunks from structural elements
        Respects boundaries and maintains context
        """
        chunks = []
        current_chunk_text = []
        current_chunk_tokens = 0
        chunk_counter = 0
        
        # Get chapter info
        chapter_name = "Unknown Chapter"
        chapter_number = 0
        topic = None
        
        if chapter_info:
            chapter_name = chapter_info.get("chapter_name", "Unknown Chapter")
            chapter_number = chapter_info.get("chapter_number", 0)
            topic = chapter_info.get("current_topic", None)
        
        for i, element in enumerate(elements):
            element_text = element['text']
            element_tokens = self._count_tokens(element_text)
            
            # Check if this is a boundary element that should start a new chunk
            should_start_new = (
                element['is_boundary'] or
                (current_chunk_tokens + element_tokens > self.chunk_size and current_chunk_text)
            )
            
            if should_start_new and current_chunk_text:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk_text)
                chunk_type = self._determine_chunk_type(current_chunk_text)
                
                chunk = ChunkMetadata(
                    chunk_id=f"p{page_num + 1}_c{chunk_counter}",
                    chunk_type=chunk_type,
                    chapter_name=chapter_name,
                    chapter_number=chapter_number,
                    page_number=page_num + 1,
                    text_content=chunk_text,
                    topic=topic
                )
                
                chunks.append(chunk)
                chunk_counter += 1
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk_text:
                    # Keep last paragraph for overlap
                    overlap_text = current_chunk_text[-1]
                    current_chunk_text = [overlap_text]
                    current_chunk_tokens = self._count_tokens(overlap_text)
                else:
                    current_chunk_text = []
                    current_chunk_tokens = 0
            
            # Add current element
            current_chunk_text.append(element_text)
            current_chunk_tokens += element_tokens
            
            # Update topic if heading detected
            if element['type'] == 'heading':
                topic = element_text.strip()
        
        # Save last chunk
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunk_type = self._determine_chunk_type(current_chunk_text)
            
            chunk = ChunkMetadata(
                chunk_id=f"p{page_num + 1}_c{chunk_counter}",
                chunk_type=chunk_type,
                chapter_name=chapter_name,
                chapter_number=chapter_number,
                page_number=page_num + 1,
                text_content=chunk_text,
                topic=topic
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(self, chunk_texts: List[str]) -> str:
        """Determine predominant type in chunk"""
        combined = '\n'.join(chunk_texts)
        return TextAnalyzer.identify_chunk_type(combined)
    
    def _create_figure_chunks(
        self,
        figures: List,
        chapter_info: Optional[Dict]
    ) -> List[ChunkMetadata]:
        """
        Create chunks for figures/diagrams
        """
        chunks = []
        
        chapter_name = "Unknown Chapter"
        chapter_number = 0
        
        if chapter_info:
            chapter_name = chapter_info.get("chapter_name", "Unknown Chapter")
            chapter_number = chapter_info.get("chapter_number", 0)
        
        for figure in figures:
            # Handle both dict and ExtractedFigure objects
            if isinstance(figure, dict):
                figure_id = figure['figure_id']
                page_number = figure['page_number']
                image_path = figure['image_path']
                caption = figure.get('caption', '')
                bbox_data = figure.get('bbox', {})
            else:
                figure_id = figure.figure_id
                page_number = figure.page_number
                image_path = figure.image_path
                caption = figure.caption or ''
                bbox_data = figure.bbox.to_dict() if figure.bbox else {}
            
            # Create text content from caption
            text_content = caption if caption else f"Figure from page {page_number}"
            
            chunk = ChunkMetadata(
                chunk_id=figure_id,
                chunk_type="diagram",
                chapter_name=chapter_name,
                chapter_number=chapter_number,
                page_number=page_number,
                text_content=text_content,
                figure_id=figure_id,
                image_path=image_path,
                caption=caption,
                bbox=BoundingBox(**bbox_data) if bbox_data else None
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_table_chunks(
        self,
        tables: Dict[int, List[Dict]],
        chapter_info: Optional[Dict]
    ) -> List[ChunkMetadata]:
        """
        Create chunks for tables
        """
        chunks = []
        
        chapter_name = "Unknown Chapter"
        chapter_number = 0
        
        if chapter_info:
            chapter_name = chapter_info.get("chapter_name", "Unknown Chapter")
            chapter_number = chapter_info.get("chapter_number", 0)
        
        for page_num, page_tables in tables.items():
            for table in page_tables:
                table_id = table['table_id']
                
                # Format table as text
                text_content = self._format_table_as_text(table)
                
                chunk = ChunkMetadata(
                    chunk_id=table_id,
                    chunk_type="table",
                    chapter_name=chapter_name,
                    chapter_number=chapter_number,
                    page_number=page_num + 1,
                    text_content=text_content
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _format_table_as_text(self, table: Dict) -> str:
        """Format table data as readable text"""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        text_lines = []
        
        if headers:
            text_lines.append(' | '.join(str(h) for h in headers if h))
            text_lines.append('-' * 50)
        
        for row in rows:
            row_text = ' | '.join(str(cell) for cell in row if cell)
            if row_text.strip():
                text_lines.append(row_text)
        
        return '\n'.join(text_lines)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))


class ChunkLinker:
    """
    Links related chunks (formulas to explanations, diagrams to text)
    """
    
    @staticmethod
    def link_diagrams_to_text(chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Link diagram chunks to nearby text chunks
        Updates related_figure_ids in text chunks
        """
        # Group by page
        chunks_by_page = {}
        for chunk in chunks:
            page = chunk.page_number
            if page not in chunks_by_page:
                chunks_by_page[page] = []
            chunks_by_page[page].append(chunk)
        
        # For each diagram, find nearby text
        for page, page_chunks in chunks_by_page.items():
            diagrams = [c for c in page_chunks if c.chunk_type == "diagram"]
            text_chunks = [c for c in page_chunks if c.chunk_type in ["text", "example"]]
            
            for diagram in diagrams:
                # Link to text chunks on same page
                for text_chunk in text_chunks:
                    if diagram.figure_id:
                        text_chunk.related_figure_ids.append(diagram.figure_id)
        
        return chunks
    
    @staticmethod
    def link_formulas(chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Identify and link related formulas
        """
        formula_chunks = [c for c in chunks if c.chunk_type == "formula"]
        
        # Simple linking: formulas on same page and topic
        for i, formula in enumerate(formula_chunks):
            for j, other_formula in enumerate(formula_chunks):
                if i != j and formula.page_number == other_formula.page_number:
                    if formula.topic == other_formula.topic:
                        formula.related_formula_ids.append(other_formula.chunk_id)
        
        return chunks


if __name__ == "__main__":
    # Example usage
    import json
    from pathlib import Path
    
    # Load extraction results
    with open("data/metadata/extraction_results.json", 'r') as f:
        extracted_content = json.load(f)
    
    # Create chunking strategy
    chunker = ChunkingStrategy(chunk_size=500, overlap=75)
    
    # Chunk document
    chapter_info = {
        "chapter_name": "Mechanical Properties of Solids",
        "chapter_number": 9,
    }
    
    chunks = chunker.chunk_document(extracted_content, chapter_info)
    
    # Link related chunks
    chunks = ChunkLinker.link_diagrams_to_text(chunks)
    chunks = ChunkLinker.link_formulas(chunks)
    
    # Save chunks
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open("data/metadata/chunks.json", 'w') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated {len(chunks)} chunks")