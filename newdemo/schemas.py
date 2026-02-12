"""
Metadata Schema and Data Models for NCERT Physics RAG System
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class BoundingBox:
    """Represents bounding box coordinates for images/figures"""
    x0: float
    y0: float
    x1: float
    y1: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        return cls(**data)


@dataclass
class ChunkMetadata:
    """
    Complete metadata schema for each chunk in the RAG system
    """
    # Required fields
    chunk_id: str
    chunk_type: str  # text, formula, diagram, table, example, definition, summary
    chapter_name: str
    chapter_number: int
    page_number: int
    text_content: str
    
    # Optional fields
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    figure_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    caption: Optional[str] = None
    image_path: Optional[str] = None
    related_formula_ids: List[str] = field(default_factory=list)
    related_figure_ids: List[str] = field(default_factory=list)
    difficulty_level: Optional[str] = None  # easy, medium, hard
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        if self.bbox:
            data['bbox'] = self.bbox.to_dict()
        # Remove embedding from dict (too large for JSON)
        data.pop('embedding', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary"""
        if 'bbox' in data and data['bbox']:
            data['bbox'] = BoundingBox.from_dict(data['bbox'])
        data.pop('embedding', None)  # Embedding stored separately
        return cls(**data)
    
    def get_search_metadata(self) -> Dict[str, Any]:
        """Get metadata for vector store search"""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "chapter_name": self.chapter_name,
            "chapter_number": self.chapter_number,
            "page_number": self.page_number,
            "topic": self.topic or "",
            "subtopic": self.subtopic or "",
            "figure_id": self.figure_id or "",
            "caption": self.caption or "",
            "image_path": self.image_path or "",
            "difficulty_level": self.difficulty_level or ""
        }


@dataclass
class Document:
    """Represents a complete NCERT Physics textbook"""
    book_title: str
    class_level: str  # "11" or "12"
    chapters: List['Chapter'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_title": self.book_title,
            "class_level": self.class_level,
            "chapters": [ch.to_dict() for ch in self.chapters]
        }
    
    def save_to_json(self, filepath: str):
        """Save document structure to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'Document':
        """Load document structure from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        doc = cls(
            book_title=data['book_title'],
            class_level=data['class_level']
        )
        doc.chapters = [Chapter.from_dict(ch) for ch in data['chapters']]
        return doc


@dataclass
class Chapter:
    """Represents a chapter in the textbook"""
    chapter_number: int
    chapter_name: str
    start_page: int
    end_page: int
    topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "chapter_name": self.chapter_name,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "topics": self.topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chapter':
        return cls(**data)


@dataclass
class ExtractedFigure:
    """Represents an extracted figure/diagram from the PDF"""
    figure_id: str
    page_number: int
    bbox: BoundingBox
    image_path: str
    caption: Optional[str] = None
    chapter_number: Optional[int] = None
    topic: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "page_number": self.page_number,
            "bbox": self.bbox.to_dict(),
            "image_path": self.image_path,
            "caption": self.caption,
            "chapter_number": self.chapter_number,
            "topic": self.topic
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedFigure':
        data['bbox'] = BoundingBox.from_dict(data['bbox'])
        return cls(**data)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with score and metadata"""
    chunk: ChunkMetadata
    score: float
    rank: int
    retrieval_method: str  # "vector", "bm25", "hybrid", "reranked"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method
        }


class MetadataValidator:
    """Validates chunk metadata against schema"""
    
    REQUIRED_FIELDS = [
        "chunk_id", "chunk_type", "chapter_name", 
        "chapter_number", "page_number", "text_content"
    ]
    
    VALID_CHUNK_TYPES = [
        "text", "formula", "diagram", "table", 
        "example", "definition", "summary", "exercise"
    ]
    
    @staticmethod
    def validate(metadata: ChunkMetadata) -> tuple[bool, List[str]]:
        """
        Validate metadata
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in MetadataValidator.REQUIRED_FIELDS:
            if not getattr(metadata, field, None):
                errors.append(f"Missing required field: {field}")
        
        # Validate chunk_type
        if metadata.chunk_type not in MetadataValidator.VALID_CHUNK_TYPES:
            errors.append(f"Invalid chunk_type: {metadata.chunk_type}")
        
        # Validate chapter_number
        if not isinstance(metadata.chapter_number, int) or metadata.chapter_number < 1:
            errors.append(f"Invalid chapter_number: {metadata.chapter_number}")
        
        # Validate page_number
        if not isinstance(metadata.page_number, int) or metadata.page_number < 1:
            errors.append(f"Invalid page_number: {metadata.page_number}")
        
        # If diagram type, should have image_path
        if metadata.chunk_type == "diagram" and not metadata.image_path:
            errors.append("Diagram chunk missing image_path")
        
        return len(errors) == 0, errors