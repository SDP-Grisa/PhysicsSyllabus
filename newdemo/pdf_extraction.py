"""
PDF Extraction Pipeline for NCERT Physics Textbooks
Handles text, formulas, diagrams, tables with noise filtering
"""

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json

from schemas import BoundingBox, ExtractedFigure, Chapter
from config import PDF_CONFIG, IMAGES_DIR


class PDFExtractor:
    """
    Multimodal PDF extraction for NCERT Physics textbooks
    """
    
    def __init__(self, pdf_path: str, output_dir: Optional[Path] = None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize both libraries
        self.fitz_doc = fitz.open(pdf_path)
        self.plumber_pdf = pdfplumber.open(pdf_path)
        
        # Storage
        self.extracted_figures: List[ExtractedFigure] = []
        self.page_texts: Dict[int, str] = {}
        self.page_tables: Dict[int, List[Dict]] = {}
        
    def extract_all(self) -> Dict[str, any]:
        """
        Main extraction pipeline
        Returns dictionary with all extracted content
        """
        print(f"Extracting from: {self.pdf_path}")
        print(f"Total pages: {len(self.fitz_doc)}")
        
        all_content = {
            "text": {},
            "figures": [],
            "tables": {},
            "metadata": {
                "total_pages": len(self.fitz_doc),
                "pdf_path": self.pdf_path
            }
        }
        
        for page_num in tqdm(range(len(self.fitz_doc)), desc="Processing pages"):
            # Extract text (cleaned)
            text = self._extract_clean_text(page_num)
            all_content["text"][page_num] = text
            
            # Extract figures
            figures = self._extract_figures(page_num)
            all_content["figures"].extend(figures)
            
            # Extract tables
            tables = self._extract_tables(page_num)
            if tables:
                all_content["tables"][page_num] = tables
        
        print(f"\nExtracted:")
        print(f"  - Pages with text: {len(all_content['text'])}")
        print(f"  - Figures: {len(all_content['figures'])}")
        print(f"  - Tables: {sum(len(t) for t in all_content['tables'].values())}")
        
        return all_content
    
    def _extract_clean_text(self, page_num: int) -> str:
        """
        Extract text from page with noise filtering
        Removes headers, footers, page numbers
        """
        page = self.plumber_pdf.pages[page_num]
        text = page.extract_text() or ""
        
        # Clean the text
        text = self._remove_headers_footers(text, page_num)
        text = self._normalize_text(text)
        
        return text
    
    def _remove_headers_footers(self, text: str, page_num: int) -> str:
        """
        Remove common header/footer patterns in NCERT textbooks
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip if line is just a page number
            if re.match(r'^\d+$', line_stripped):
                continue
            
            # Skip if line looks like header (all caps, short)
            if line_stripped.isupper() and len(line_stripped) < 50:
                # But keep if it's likely a section heading with number
                if not re.match(r'^\d+\.?\d*\s+[A-Z]', line_stripped):
                    continue
            
            # Skip common NCERT footer patterns
            if any(pattern in line_stripped.lower() for pattern in [
                'physics', 'ncert', 'class xi', 'class xii', 
                'part i', 'part ii', 'chapter'
            ]) and len(line_stripped) < 40:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize spacing and formatting"""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix common OCR issues
        text = text.replace('–', '-').replace('—', '-')
        return text.strip()
    
    def _extract_figures(self, page_num: int) -> List[ExtractedFigure]:
        """
        Extract figures/diagrams from page using bounding box detection
        """
        figures = []
        fitz_page = self.fitz_doc[page_num]
        
        # Get images from page
        image_list = fitz_page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            # Get image bbox
            img_rects = fitz_page.get_image_rects(xref)
            
            if not img_rects:
                continue
            
            for rect in img_rects:
                # Check minimum size
                width = rect.width
                height = rect.height
                
                if width < PDF_CONFIG["min_figure_width"] or \
                   height < PDF_CONFIG["min_figure_height"]:
                    continue
                
                # Extract and save image
                try:
                    base_image = self.fitz_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Generate unique figure ID
                    figure_id = self._generate_figure_id(page_num, img_index)
                    image_filename = f"{figure_id}.png"
                    image_path = self.output_dir / image_filename
                    
                    # Save image
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    img_obj.save(image_path, "PNG")
                    
                    # Create ExtractedFigure object
                    bbox = BoundingBox(
                        x0=rect.x0,
                        y0=rect.y0,
                        x1=rect.x1,
                        y1=rect.y1
                    )
                    
                    # Try to extract caption (text near image)
                    caption = self._extract_caption_near_image(fitz_page, rect)
                    
                    figure = ExtractedFigure(
                        figure_id=figure_id,
                        page_number=page_num + 1,  # 1-indexed
                        bbox=bbox,
                        image_path=str(image_path),
                        caption=caption
                    )
                    
                    figures.append(figure)
                    
                except Exception as e:
                    print(f"Error extracting image on page {page_num}: {e}")
                    continue
        
        return figures
    
    def _extract_caption_near_image(self, page: fitz.Page, img_rect: fitz.Rect) -> Optional[str]:
        """
        Extract caption text near an image
        Looks below the image for "Fig X.Y" patterns
        """
        # Define search area (below image)
        search_rect = fitz.Rect(
            img_rect.x0,
            img_rect.y1,
            img_rect.x1,
            img_rect.y1 + 100  # 100 points below
        )
        
        # Extract text in this region
        text = page.get_textbox(search_rect)
        
        if not text:
            return None
        
        # Look for figure caption patterns
        caption_match = re.search(
            r'(Fig\.?\s*\d+\.?\d*.*?)(?:\n|$)',
            text,
            re.IGNORECASE
        )
        
        if caption_match:
            return caption_match.group(1).strip()
        
        return None
    
    def _extract_tables(self, page_num: int) -> List[Dict]:
        """
        Extract tables from page using pdfplumber
        """
        page = self.plumber_pdf.pages[page_num]
        tables = page.extract_tables()
        
        formatted_tables = []
        
        for table_idx, table in enumerate(tables or []):
            if not table or len(table) < 2:
                continue
            
            # Format table as dict
            formatted_table = {
                "table_id": f"page{page_num + 1}_table{table_idx + 1}",
                "headers": table[0] if table else [],
                "rows": table[1:] if len(table) > 1 else [],
                "raw_data": table
            }
            
            formatted_tables.append(formatted_table)
        
        return formatted_tables
    
    def _generate_figure_id(self, page_num: int, img_index: int) -> str:
        """Generate unique figure ID"""
        return f"fig_p{page_num + 1:03d}_i{img_index:02d}"
    
    def close(self):
        """Close PDF resources"""
        self.fitz_doc.close()
        self.plumber_pdf.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TextAnalyzer:
    """
    Analyzes extracted text to identify structure
    Detects: headings, examples, formulas, definitions
    """
    
    # Patterns for structure detection
    HEADING_PATTERN = r'^(\d+\.?\d*)\s+([A-Z][A-Za-z\s]+)$'
    EXAMPLE_PATTERN = r'(?:Example|EXAMPLE)\s*\d*\.?\d*'
    FORMULA_PATTERN = r'[A-Za-z]\s*=\s*[^.]+(?:\n|$)'
    DEFINITION_PATTERN = r'^(?:Definition|DEFINITION)[\s:]'
    
    @staticmethod
    def identify_chunk_type(text: str) -> str:
        """
        Identify the type of text chunk
        Returns: text, formula, example, definition, summary
        """
        text_upper = text.strip().upper()
        
        # Check for example
        if re.search(TextAnalyzer.EXAMPLE_PATTERN, text):
            return "example"
        
        # Check for definition
        if re.search(TextAnalyzer.DEFINITION_PATTERN, text):
            return "definition"
        
        # Check for summary
        if 'SUMMARY' in text_upper and len(text) < 500:
            return "summary"
        
        # Check for formula (high density of math symbols)
        math_symbols = len(re.findall(r'[=∫∂Σ√∞≈±×÷]', text))
        if math_symbols > 3 or (math_symbols > 0 and len(text) < 100):
            return "formula"
        
        # Default to text
        return "text"
    
    @staticmethod
    def extract_headings(text: str) -> List[Tuple[int, str]]:
        """
        Extract section headings with hierarchy
        Returns list of (level, heading_text)
        """
        lines = text.split('\n')
        headings = []
        
        for line in lines:
            line = line.strip()
            match = re.match(TextAnalyzer.HEADING_PATTERN, line)
            
            if match:
                number = match.group(1)
                heading = match.group(2)
                
                # Determine level from numbering (1.1 vs 1.1.1)
                level = number.count('.')
                headings.append((level, heading))
        
        return headings
    
    @staticmethod
    def detect_formula_context(text: str, window: int = 100) -> List[Tuple[str, str]]:
        """
        Detect formulas and their surrounding context
        Returns list of (formula, context)
        """
        formulas = []
        matches = re.finditer(TextAnalyzer.FORMULA_PATTERN, text)
        
        for match in matches:
            formula = match.group(0).strip()
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end]
            
            formulas.append((formula, context))
        
        return formulas


def save_extraction_results(content: Dict, output_path: Path):
    """Save extraction results to JSON"""
    # Convert ExtractedFigure objects to dicts
    if 'figures' in content:
        content['figures'] = [
            fig.to_dict() if hasattr(fig, 'to_dict') else fig
            for fig in content['figures']
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    
    print(f"Extraction results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    pdf_path = "data/pdfs/physics_class11.pdf"  # Update with actual path
    
    with PDFExtractor(pdf_path) as extractor:
        content = extractor.extract_all()
        save_extraction_results(content, Path("data/metadata/extraction_results.json"))