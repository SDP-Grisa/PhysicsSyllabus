"""
Improved PDF Content Extractor with Topic-wise and Paragraph-wise Extraction
"""

import pdfplumber
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import hashlib


class ImprovedPDFExtractor:
    """
    Enhanced extractor with:
    - Better paragraph detection
    - Topic/section-based chunking
    - Hierarchical structure preservation
    - Context-aware chunking
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = Path("extracted_content")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
    def _is_header_footer(self, text: str, y_position: float = 0, page_height: float = 1000) -> bool:
        """Detect header/footer content"""
        if not text or len(text.strip()) < 3:
            return True
            
        text_stripped = text.strip()
        
        # Common header/footer patterns
        patterns = [
            r'^\d+\s*$',
            r'^[A-Z\s]+\d+$',
            r'Reprint\s+\d{4}-\d{2}',
            r'^NCERT',
            r'©\s*NCERT',
            r'Not to be republished',
            r'^\s*PHYSICS\s*$',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_stripped, re.IGNORECASE):
                return True
        return False
    
    def _detect_section_heading(self, text: str) -> Optional[Dict]:
        """
        Detect section headings like:
        - 8.1 INTRODUCTION
        - 8.2 ELASTIC BEHAVIOUR OF SOLIDS
        """
        heading_patterns = [
            r'^(\d+)\.(\d+)\s+([A-Z][A-Z\s]+)$',  # 8.1 INTRODUCTION
            r'^(\d+)\.(\d+)\.(\d+)\s+([A-Z][A-Z\s]+)$',  # 8.1.1 SUBSECTION
        ]
        
        for pattern in heading_patterns:
            match = re.match(pattern, text.strip())
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    return {
                        'type': 'section',
                        'chapter': int(groups[0]),
                        'section': int(groups[1]),
                        'title': groups[2].strip(),
                        'full_number': f"{groups[0]}.{groups[1]}"
                    }
                elif len(groups) == 4:
                    return {
                        'type': 'subsection',
                        'chapter': int(groups[0]),
                        'section': int(groups[1]),
                        'subsection': int(groups[2]),
                        'title': groups[3].strip(),
                        'full_number': f"{groups[0]}.{groups[1]}.{groups[2]}"
                    }
        
        return None
    
    def _detect_equation(self, text: str) -> bool:
        """Detect mathematical equations"""
        equation_patterns = [
            r'[=+\-*/^√∫∑∏]',
            r'\b\w+\s*=\s*\w+',
            r'\d+\s*[×÷]\s*\d+',
            r'[α-ωΑ-Ω]',
            r'\b(?:sin|cos|tan|log|ln|exp)\b',
            r'\^\d+',
            r'\([^)]+\)',
        ]
        
        match_count = sum(1 for pattern in equation_patterns if re.search(pattern, text))
        return match_count >= 2
    
    def _detect_problem(self, text: str) -> bool:
        """Detect problem/sum/exercise"""
        problem_indicators = [
            r'^(?:Example|EXAMPLE)\s+\d+',
            r'^(?:Problem|PROBLEM)\s+\d+',
            r'^(?:Exercise|EXERCISE)\s+\d+',
            r'^\d+\.\s*(?:Find|Calculate|Determine|Prove|Show|Derive)',
            r'Solution:',
            r'Given:',
        ]
        
        return any(re.search(pattern, text.strip(), re.MULTILINE) for pattern in problem_indicators)
    
    def _extract_chapter_info(self, text: str) -> Optional[Dict]:
        """Extract chapter number and title"""
        chapter_patterns = [
            r'CHAPTER\s+(EIGHT|NINE|TEN|ELEVEN|TWELVE|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN)\s*\n\s*([^\n]+)',
            r'CHAPTER\s+(\d+)\s*\n\s*([^\n]+)',
        ]
        
        word_to_num = {
            'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
            'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10,
            'ELEVEN': 11, 'TWELVE': 12
        }
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                ch_num_str = match.group(1).upper()
                ch_title = match.group(2).strip()
                
                if ch_num_str in word_to_num:
                    ch_num = word_to_num[ch_num_str]
                else:
                    ch_num = int(ch_num_str)
                
                return {
                    'chapter_number': ch_num,
                    'chapter_title': ch_title
                }
        
        return None
    
    def _clean_and_segment_text(self, text: str) -> List[str]:
        """
        Clean text and segment into meaningful paragraphs
        Preserves paragraph structure better than simple splitting
        """
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split into lines
        lines = text.split('\n')
        
        paragraphs = []
        current_para = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                continue
            
            # Skip headers/footers
            if self._is_header_footer(line):
                continue
            
            # Check if this is a new paragraph indicator
            # (starts with capital, previous ended with period, or is a heading)
            is_new_para = (
                len(line) > 0 and line[0].isupper() and
                (not current_para or current_para[-1].endswith('.') or 
                 current_para[-1].endswith('?') or current_para[-1].endswith('!'))
            )
            
            if is_new_para and current_para and len(' '.join(current_para)) > 50:
                paragraphs.append(' '.join(current_para))
                current_para = [line]
            else:
                current_para.append(line)
        
        # Add final paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Filter out very short paragraphs (likely noise)
        paragraphs = [p for p in paragraphs if len(p.strip()) > 30]
        
        return paragraphs
    
    def _create_contextual_chunks(self, paragraphs: List[str], 
                                  current_section: Optional[Dict],
                                  current_chapter: Optional[Dict],
                                  page_num: int) -> List[Dict]:
        """
        Create chunks with contextual information
        Combines related paragraphs and adds hierarchical context
        """
        chunks = []
        
        # Group related paragraphs (max 3 paragraphs per chunk for coherence)
        for i in range(0, len(paragraphs), 2):  # Overlap by taking 2 at a time
            para_group = paragraphs[i:min(i+3, len(paragraphs))]
            combined_text = ' '.join(para_group)
            
            # Skip if too short
            if len(combined_text) < 50:
                continue
            
            # Determine content type
            content_type = 'text'
            if self._detect_equation(combined_text):
                content_type = 'equation'
            elif self._detect_problem(combined_text):
                content_type = 'problem'
            
            # Create chunk with rich context
            chunk = {
                'page': page_num,
                'content': combined_text,
                'type': content_type,
                'chapter': current_chapter,
                'section': current_section,
                'context': self._generate_context_string(current_chapter, current_section)
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _generate_context_string(self, chapter: Optional[Dict], 
                                 section: Optional[Dict]) -> str:
        """
        Generate a context string for better retrieval
        This helps LLM understand where the content comes from
        """
        context_parts = []
        
        if chapter:
            context_parts.append(f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}")
        
        if section:
            section_title = section.get('title', '')
            if section.get('type') == 'section':
                context_parts.append(f"Section {section['full_number']}: {section_title}")
            elif section.get('type') == 'subsection':
                context_parts.append(f"Subsection {section['full_number']}: {section_title}")
        
        return ' | '.join(context_parts) if context_parts else ''
    
    def extract_images_from_page(self, page_num: int, has_figures: bool = False) -> List[Dict]:
        """Extract images from a specific page"""
        images_data = []
        
        if not has_figures:
            return images_data
        
        try:
            images = convert_from_path(
                self.pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=150
            )
            
            if images:
                img = images[0]
                
                img_hash = hashlib.md5(f"page_{page_num}".encode()).hexdigest()[:8]
                img_path = self.output_dir / "images" / f"page_{page_num}_{img_hash}.png"
                img.save(img_path)
                
                try:
                    ocr_text = pytesseract.image_to_string(img)
                except:
                    ocr_text = ""
                
                images_data.append({
                    'page': page_num,
                    'path': str(img_path).replace('\\', '/'),
                    'ocr_text': ocr_text.strip(),
                    'type': 'page_image'
                })
        
        except Exception as e:
            print(f"Error extracting image from page {page_num}: {e}")
        
        return images_data
    
    def extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables_data = []
        
        try:
            tables = page.extract_tables()
            
            for idx, table in enumerate(tables):
                if table and len(table) > 1:
                    has_content = any(
                        any(cell and str(cell).strip() for cell in row)
                        for row in table
                    )
                    
                    if not has_content:
                        continue
                    
                    headers = table[0] if table[0] else [f"Col_{i}" for i in range(len(table[0]))]
                    rows = table[1:]
                    
                    table_dict = {
                        'page': page_num,
                        'headers': headers,
                        'rows': rows,
                        'markdown': self._table_to_markdown(headers, rows)
                    }
                    
                    table_file = self.output_dir / "tables" / f"table_page{page_num}_idx{idx}.json"
                    with open(table_file, 'w', encoding='utf-8') as f:
                        json.dump(table_dict, f, indent=2, ensure_ascii=False)
                    
                    tables_data.append(table_dict)
        
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {e}")
        
        return tables_data
    
    def _table_to_markdown(self, headers: List, rows: List[List]) -> str:
        """Convert table to markdown format"""
        if not headers or not rows:
            return ""
        
        headers = [str(h) if h else "" for h in headers]
        
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in rows:
            clean_row = [str(cell) if cell else "" for cell in row]
            md += "| " + " | ".join(clean_row) + " |\n"
        
        return md
    
    def extract_all_content(self) -> Dict:
        """Main extraction method with improved structure"""
        all_content = {
            'text_chunks': [],
            'equations': [],
            'problems': [],
            'tables': [],
            'images': [],
            'chapters': [],
            'sections': []
        }
        
        current_chapter = None
        current_section = None
        
        print("Starting improved PDF extraction...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_num}/{total_pages}...")
                
                text = page.extract_text() or ""
                
                # Check for chapter
                chapter_info = self._extract_chapter_info(text)
                if chapter_info:
                    current_chapter = chapter_info
                    current_section = None  # Reset section when new chapter starts
                    all_content['chapters'].append(chapter_info)
                    print(f"Found Chapter {chapter_info['chapter_number']}: {chapter_info['chapter_title']}")
                
                # Check for section headings
                lines = text.split('\n')
                for line in lines:
                    section_info = self._detect_section_heading(line)
                    if section_info:
                        current_section = section_info
                        all_content['sections'].append(section_info)
                        print(f"  Found {section_info['type']}: {section_info['full_number']} - {section_info['title']}")
                
                # Extract and segment text into paragraphs
                paragraphs = self._clean_and_segment_text(text)
                
                # Create contextual chunks
                chunks = self._create_contextual_chunks(
                    paragraphs, 
                    current_section, 
                    current_chapter, 
                    page_num
                )
                
                for chunk in chunks:
                    all_content['text_chunks'].append(chunk)
                    
                    if chunk['type'] == 'equation':
                        all_content['equations'].append(chunk)
                    elif chunk['type'] == 'problem':
                        all_content['problems'].append(chunk)
                
                # Extract tables
                tables = self.extract_tables_from_page(page, page_num)
                for table in tables:
                    table['chapter'] = current_chapter
                    table['section'] = current_section
                    table['context'] = self._generate_context_string(current_chapter, current_section)
                    all_content['tables'].append(table)
                
                # Extract images
                has_figures = bool(
                    re.search(r'Fig\.|Figure|Diagram|Graph|Table\s+\d+\.\d+', text, re.IGNORECASE)
                )
                
                if has_figures:
                    images = self.extract_images_from_page(page_num, has_figures=True)
                    for img in images:
                        img['chapter'] = current_chapter
                        img['section'] = current_section
                        img['context'] = self._generate_context_string(current_chapter, current_section)
                        all_content['images'].append(img)
        
        print(f"\nExtraction complete!")
        print(f"Text chunks: {len(all_content['text_chunks'])}")
        print(f"Equations: {len(all_content['equations'])}")
        print(f"Problems: {len(all_content['problems'])}")
        print(f"Tables: {len(all_content['tables'])}")
        print(f"Images: {len(all_content['images'])}")
        print(f"Chapters: {len(all_content['chapters'])}")
        print(f"Sections: {len(all_content['sections'])}")
        
        # Save to JSON
        output_file = self.output_dir / "extracted_content.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, indent=2, ensure_ascii=False)
        
        print(f"\nContent saved to {output_file}")
        
        return all_content


if __name__ == "__main__":
    pdf_path = "../pdf/keph201.pdf"
    
    extractor = ImprovedPDFExtractor(pdf_path)
    content = extractor.extract_all_content()