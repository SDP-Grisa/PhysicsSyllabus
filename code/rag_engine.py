"""
LLM-Integrated RAG Query Engine with Groq API
Provides accurate, context-aware responses without hallucination
"""

import re
import os
import json
from typing import Dict, List, Optional
from vector_db import MultimodalVectorDB
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMIntegratedRAGEngine:
    def __init__(self, db: MultimodalVectorDB, content_file: str, api_key: Optional[str] = None):
        self.db = db
        with open(content_file, 'r', encoding='utf-8') as f:
            self.content = json.load(f)
        self.chapters = {ch['chapter_number']: ch for ch in self.content['chapters']}
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq API
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.llm_enabled = True
                print("✅ LLM integration enabled with Groq API (Llama 3.3 70B)")
            except Exception as e:
                print(f"⚠️ Groq initialization failed: {e}")
                self.client = None
                self.llm_enabled = False
        else:
            self.llm_enabled = False
            self.client = None
            print("⚠️  LLM integration disabled (no GROQ_API_KEY). Using basic retrieval only.")
        
        # Intent descriptions
        self.intent_descriptions = {
            'summary': "The query is asking for a summary, overview, brief explanation, or key points",
            'cross_chapter': "The query is asking about multiple chapters, across the book, or which chapters cover a topic",
            'problem': "The query is asking for problems, examples, exercises, solutions, or calculations",
            'detailed_explanation': "The query asks for detailed description, explanation, or in-depth coverage of a physics concept"
        }
        
        self.content_type_descriptions = {
            'equation': "The query is about equations, formulas, derivations, or mathematical expressions",
            'table': "The query is about tables, data values, comparisons, or lists of properties",
            'image': "The query is about diagrams, figures, illustrations, graphs, or visual representations"
        }
        
        # Precompute embeddings
        self.intent_embs = {k: self.embedder.encode([v])[0] for k, v in self.intent_descriptions.items()}
        self.content_type_embs = {k: self.embedder.encode([v])[0] for k, v in self.content_type_descriptions.items()}

    def parse_query(self, query: str) -> Dict:
        """Parse and understand the query"""
        query_lower = query.lower()
        query_emb = self.embedder.encode([query])[0]
        
        parsed = {
            'original_query': query,
            'query_type': 'general',
            'requires_summary': False,
            'requires_cross_chapter': False,
            'content_types': ['text'],
        }
        
        # Intent detection
        sims = {k: cosine_similarity([query_emb], [v])[0][0] for k, v in self.intent_embs.items()}
        best_intent = max(sims, key=sims.get, default='general')
        if sims[best_intent] > 0.7:
            parsed['query_type'] = best_intent
            if best_intent == 'summary':
                parsed['requires_summary'] = True
            if best_intent == 'cross_chapter':
                parsed['requires_cross_chapter'] = True
        
        # Fallback keywords
        if sims[best_intent] < 0.7:
            if any(w in query_lower for w in ['summary', 'summarize', 'overview', 'explain']):
                parsed['query_type'] = 'summary'
                parsed['requires_summary'] = True
            if any(p in query_lower for p in ['how many chapters', 'which chapters', 'all chapters', 'across chapters']):
                parsed['query_type'] = 'cross_chapter'
                parsed['requires_cross_chapter'] = True
            if any(w in query_lower for w in ['problem', 'example', 'solve', 'sum', 'exercise', 'question']):
                parsed['query_type'] = 'problem'
        
        # Content type detection
        for ct, emb in self.content_type_embs.items():
            if cosine_similarity([query_emb], [emb])[0][0] > 0.6:
                if ct not in parsed['content_types']:
                    parsed['content_types'].append(ct)
        
        return parsed

    def execute_query(self, query: str, max_results: int = 10) -> Dict:
        """Execute query and generate LLM response"""
        parsed = self.parse_query(query)
        
        # Get relevant content based on query type
        if parsed['query_type'] == 'cross_chapter':
            retrieval_results = self._handle_cross_chapter_query(parsed, max_results)
        elif parsed['query_type'] == 'summary':
            retrieval_results = self._handle_summary_query(parsed, max_results)
        elif parsed['query_type'] == 'problem':
            retrieval_results = self._handle_problem_query(parsed, max_results)
        else:
            retrieval_results = self._handle_general_query(parsed, max_results)
        
        # Generate LLM response if enabled
        if self.llm_enabled:
            llm_response = self._generate_llm_response(query, retrieval_results, parsed)
            retrieval_results['llm_response'] = llm_response
        
        return retrieval_results

    def _handle_general_query(self, parsed: Dict, max_results: int) -> Dict:
        """Handle general queries - retrieve ALL relevant multimodal content"""
        results = {
            'query_type': 'general',
            'query': parsed['original_query'],
            'results': {}
        }
        
        # ALWAYS retrieve text content
        text_results = self.db.search_text(
            parsed['original_query'], max_results
        )
        results['results']['text'] = text_results
        
        # ALWAYS try to find related equations, tables, and images
        equations = self.db.search_equations(
            parsed['original_query'], 6
        )
        if equations:
            results['results']['equations'] = equations
        
        tables = self.db.search_tables(
            parsed['original_query'], 5
        )
        if tables:
            results['results']['tables'] = tables
        
        images = self.db.search_images(
            parsed['original_query'], 6
        )
        if images:
            results['results']['images'] = images
        
        # Also check for related problems/examples
        problems = self.db.search_problems(
            parsed['original_query'], 5
        )
        if problems:
            results['results']['problems'] = problems
        
        # SMART ENHANCEMENT: If we found text content, also search for content
        # from the same chapter/section to get complete context
        if text_results:
            # Get chapter from top result
            top_chapter = text_results[0].get('metadata', {}).get('chapter_number')
            
            if top_chapter:
                # Search for additional diagrams/tables from same chapter
                chapter_images = self.db.search_images(
                    parsed['original_query'], 3, chapter_filter=top_chapter
                )
                if chapter_images and 'images' not in results['results']:
                    results['results']['images'] = chapter_images
                elif chapter_images and 'images' in results['results']:
                    # Merge and deduplicate
                    existing_ids = {img.get('id') for img in results['results']['images']}
                    for img in chapter_images:
                        if img.get('id') not in existing_ids:
                            results['results']['images'].append(img)
        
        return results

    def _handle_summary_query(self, parsed: Dict, max_results: int) -> Dict:
        """Handle summary requests"""
        results = {
            'query_type': 'summary',
            'query': parsed['original_query'],
            'results': {}
        }
        
        # Topic summary across all chapters
        all_results = self.db.search_all(
            parsed['original_query']
        )
        results['results'] = all_results
        
        return results
    
    def _handle_cross_chapter_query(self, parsed: Dict, max_results: int) -> Dict:
        """Handle cross-chapter queries"""
        results = {
            'query_type': 'cross_chapter',
            'query': parsed['original_query'],
            'results': {}
        }
        
        all_results = self.db.search_text(
            parsed['original_query'],
            n_results=50
        )
        
        chapters_found = {}
        for result in all_results:
            chapter_num = result['metadata'].get('chapter_number')
            if chapter_num:
                if chapter_num not in chapters_found:
                    chapters_found[chapter_num] = {
                        'chapter': self.chapters.get(chapter_num),
                        'relevant_content': []
                    }
                chapters_found[chapter_num]['relevant_content'].append(result)
        
        results['results']['chapters'] = chapters_found
        results['results']['chapter_count'] = len(chapters_found)
        
        return results
    
    def _handle_problem_query(self, parsed: Dict, max_results: int) -> Dict:
        """Handle problem queries - include equations, diagrams, and tables"""
        results = {
            'query_type': 'problem',
            'query': parsed['original_query'],
            'results': {}
        }
        
        # Search problems
        problems = self.db.search_problems(
            parsed['original_query'],
            n_results=max_results
        )
        results['results']['problems'] = problems
        
        # ALWAYS get related equations (essential for problem-solving)
        equations = self.db.search_equations(
            parsed['original_query'],
            n_results=6
        )
        if equations:
            results['results']['equations'] = equations
        
        # Also get diagrams (often helpful for problem visualization)
        images = self.db.search_images(
            parsed['original_query'],
            n_results=3
        )
        if images:
            results['results']['images'] = images
        
        # Get tables (for reference data)
        tables = self.db.search_tables(
            parsed['original_query'],
            n_results=3
        )
        if tables:
            results['results']['tables'] = tables
        
        # Also include some explanatory text
        text = self.db.search_text(
            parsed['original_query'],
            n_results=5
        )
        if text:
            results['results']['text'] = text
        
        return results
    
    def _generate_llm_response(self, query: str, retrieval_results: Dict, parsed: Dict) -> str:
        """
        Generate accurate LLM response based on retrieved content
        Prevents hallucination by grounding in retrieved documents
        """
        if not self.client:
            return "AI response generation is disabled (missing or invalid Groq API key). Showing raw retrieved content instead."
        
        try:
            # Prepare context from retrieved results
            context = self._prepare_context_for_llm(retrieval_results)
            
            if not context.strip():
                return "I couldn't find relevant information in the textbook to answer your question. Please try rephrasing or asking about a different topic."
            
            # Create system prompt
            system_prompt = """You are a helpful physics tutor assisting students with NCERT Physics (Class 11) textbook questions. 

Your role is to:
1. Answer questions ONLY based on the provided textbook content
2. NEVER make up information or add facts not present in the context
3. If the context doesn't contain enough information, explicitly say so
4. Cite chapter/section numbers when relevant
5. Explain concepts clearly and accurately
6. Use the exact formulas and definitions from the textbook
7. For numerical problems, show step-by-step solutions using textbook methods
8. When diagrams/figures are available in the context, MENTION them and explain what they show
9. When tables are available, reference them in your explanation

CRITICAL: Only use information from the provided context. If you're not sure about something, say you don't have that information rather than guessing.

When diagrams are listed, tell the user to "see the diagram below" or "refer to Figure X shown below" to connect your explanation with the visual aids."""

            # Create user prompt
            user_prompt = f"""Based on the following content from the NCERT Physics textbook, please answer this question:

QUESTION: {query}

TEXTBOOK CONTENT:
{context}

Please provide a clear, accurate answer based ONLY on the textbook content above. If the content doesn't fully answer the question, acknowledge what information is missing."""

            # Call Groq API with Llama 3.3 70B
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3,  # Low temperature for more factual responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _prepare_context_for_llm(self, retrieval_results: Dict) -> str:
        """
        Prepare retrieved content as context for LLM
        Organizes content by type and relevance
        """
        context_parts = []
        
        results = retrieval_results.get('results', {})
        
        # Add text content
        if 'text' in results and results['text']:
            context_parts.append("=== TEXTBOOK PARAGRAPHS ===")
            for idx, item in enumerate(results['text'][:8], 1):  # Top 8 results
                metadata = item.get('metadata', {})
                context_str = f"\n[Source {idx}]"
                
                if metadata.get('chapter_number'):
                    context_str += f" Chapter {metadata['chapter_number']}"
                    if metadata.get('chapter_title'):
                        context_str += f": {metadata['chapter_title']}"
                
                if item.get('context'):
                    context_str += f" | {item['context']}"
                
                context_str += f" (Page {metadata.get('page', 'N/A')})"
                context_str += f"\n{item.get('content', '')}\n"
                
                context_parts.append(context_str)
        
        # Add equations
        if 'equations' in results and results['equations']:
            context_parts.append("\n=== RELEVANT EQUATIONS/FORMULAS ===")
            for idx, item in enumerate(results['equations'][:5], 1):
                metadata = item.get('metadata', {})
                context_str = f"\n[Equation {idx}]"
                
                if metadata.get('chapter_number'):
                    context_str += f" Chapter {metadata['chapter_number']}"
                
                if item.get('context'):
                    context_str += f" | {item['context']}"
                
                context_str += f"\n{item.get('content', '')}\n"
                context_parts.append(context_str)
        
        # Add problems/examples
        if 'problems' in results and results['problems']:
            context_parts.append("\n=== WORKED EXAMPLES ===")
            for idx, item in enumerate(results['problems'][:3], 1):
                metadata = item.get('metadata', {})
                context_str = f"\n[Example {idx}]"
                
                if metadata.get('chapter_number'):
                    context_str += f" Chapter {metadata['chapter_number']}"
                
                if item.get('context'):
                    context_str += f" | {item['context']}"
                
                context_str += f"\n{item.get('content', '')}\n"
                context_parts.append(context_str)
        
        # Add tables
        if 'tables' in results and results['tables']:
            context_parts.append("\n=== DATA TABLES ===")
            for idx, item in enumerate(results['tables'][:3], 1):
                metadata = item.get('metadata', {})
                table_data = item.get('table_data', {})
                
                context_str = f"\n[Table {idx}]"
                if metadata.get('chapter_number'):
                    context_str += f" Chapter {metadata['chapter_number']}"
                
                if item.get('context'):
                    context_str += f" | {item['context']}"
                
                context_str += f"\n{table_data.get('markdown', '')}\n"
                context_parts.append(context_str)
        
        # Add information about available diagrams/figures
        if 'images' in results and results['images']:
            context_parts.append("\n=== AVAILABLE DIAGRAMS/FIGURES ===")
            for idx, item in enumerate(results['images'][:5], 1):
                metadata = item.get('metadata', {})
                image_data = item.get('image_data', {})
                
                context_str = f"\n[Diagram {idx}]"
                if metadata.get('chapter_number'):
                    context_str += f" Chapter {metadata['chapter_number']}"
                
                if item.get('context'):
                    context_str += f" | {item['context']}"
                
                context_str += f" (Page {metadata.get('page', 'N/A')})"
                
                # Include OCR text if available
                ocr_text = image_data.get('ocr_text', '')
                if ocr_text:
                    context_str += f"\nDiagram content/labels: {ocr_text[:200]}"
                else:
                    context_str += f"\nDiagram available for this topic"
                
                context_str += "\n"
                context_parts.append(context_str)
        
        # Add chapter summary for cross-chapter queries
        if 'chapters' in results:
            context_parts.append("\n=== CHAPTERS CONTAINING RELEVANT INFORMATION ===")
            for ch_num, ch_data in results['chapters'].items():
                ch_info = ch_data.get('chapter', {})
                context_parts.append(f"\nChapter {ch_num}: {ch_info.get('chapter_title', '')}")
                
                # Add top relevant content from each chapter
                for content in ch_data.get('relevant_content', [])[:2]:
                    context_parts.append(f"  - {content.get('content', '')[:200]}...")
        
        return '\n'.join(context_parts)
    
    def get_chapter_list(self) -> List[Dict]:
        """Get list of all chapters"""
        return sorted(self.chapters.values(), key=lambda x: x['chapter_number'])


class ResponseFormatter:
    """Format RAG results for display"""
    
    @staticmethod
    def format_text_result(result: Dict) -> str:
        """Format text result with metadata"""
        metadata = result['metadata']
        chapter_info = ""
        
        if metadata.get('chapter_number'):
            chapter_info = f"**Chapter {metadata['chapter_number']}: {metadata.get('chapter_title', '')}** | Page {metadata.get('page', 'N/A')}"
        
        # Add context if available
        if result.get('context'):
            chapter_info += f"\n*Context: {result['context']}*"
        
        return f"{chapter_info}\n\n{result['content']}\n"
    
    @staticmethod
    def format_equation_result(result: Dict) -> str:
        """Format equation"""
        metadata = result['metadata']
        chapter_info = ""
        
        if metadata.get('chapter_number'):
            chapter_info = f"**Chapter {metadata['chapter_number']}** | Page {metadata.get('page', 'N/A')}"
        
        if result.get('context'):
            chapter_info += f"\n*Context: {result['context']}*"
        
        equation = result['content']
        
        return f"{chapter_info}\n\n```\n{equation}\n```\n"
    
    @staticmethod
    def format_problem_result(result: Dict) -> str:
        """Format problem"""
        metadata = result['metadata']
        chapter_info = ""
        
        if metadata.get('chapter_number'):
            chapter_info = f"**Example from Chapter {metadata['chapter_number']}** | Page {metadata.get('page', 'N/A')}"
        
        if result.get('context'):
            chapter_info += f"\n*Context: {result['context']}*"
        
        return f"{chapter_info}\n\n{result['content']}\n"
    
    @staticmethod
    def format_table_result(result: Dict) -> str:
        """Format table"""
        table_data = result.get('table_data', {})
        metadata = result.get('metadata', {})
        
        chapter_info = ""
        if metadata.get('chapter_number'):
            chapter_info = f"**Table from Chapter {metadata['chapter_number']}** | Page {metadata.get('page', 'N/A')}\n\n"
        
        markdown_table = table_data.get('markdown', '')
        
        return f"{chapter_info}{markdown_table}\n"
    
    @staticmethod
    def format_image_result(result: Dict) -> Dict:
        """Format image"""
        image_data = result.get('image_data', {})
        metadata = result.get('metadata', {})
        
        return {
            'chapter': metadata.get('chapter_number'),
            'page': metadata.get('page'),
            'image_path': image_data.get('path'),
            'ocr_text': image_data.get('ocr_text', ''),
            'context': result.get('context', '')
        }
    
    @staticmethod
    def format_complete_response(query_results: Dict) -> Dict:
        """Format complete query response including LLM response"""
        formatted = {
            'query': query_results['query'],
            'query_type': query_results['query_type'],
            'llm_response': query_results.get('llm_response', ''),
            'formatted_results': {}
        }
        
        results = query_results.get('results', {})
        
        if 'text' in results:
            formatted['formatted_results']['text'] = [
                ResponseFormatter.format_text_result(r) for r in results['text']
            ]
        
        if 'equations' in results:
            formatted['formatted_results']['equations'] = [
                ResponseFormatter.format_equation_result(r) for r in results['equations']
            ]
        
        if 'problems' in results:
            formatted['formatted_results']['problems'] = [
                ResponseFormatter.format_problem_result(r) for r in results['problems']
            ]
        
        if 'tables' in results:
            formatted['formatted_results']['tables'] = [
                ResponseFormatter.format_table_result(r) for r in results['tables']
            ]
        
        if 'images' in results:
            formatted['formatted_results']['images'] = [
                ResponseFormatter.format_image_result(r) for r in results['images']
            ]
        
        if 'chapters' in results:
            formatted['formatted_results']['chapters'] = results['chapters']
            formatted['formatted_results']['chapter_count'] = results.get('chapter_count', 0)
        
        return formatted