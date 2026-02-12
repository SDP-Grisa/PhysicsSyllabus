"""
LLM Answer Generation with Grounded, Hallucination-Free Responses
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Optional, Dict, Tuple
import re

from schemas import RetrievalResult
from config import LLM_CONFIG, SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class AnswerGenerator:
    """
    Generates grounded answers using open-source LLMs
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or LLM_CONFIG["model_name"]
        self.device = LLM_CONFIG["device"]
        
        print(f"Loading LLM: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        if LLM_CONFIG["load_in_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            quantization_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        print("LLM loaded successfully")
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=LLM_CONFIG["max_new_tokens"],
            temperature=LLM_CONFIG["temperature"],
            top_p=LLM_CONFIG["top_p"],
            do_sample=LLM_CONFIG["do_sample"],
            repetition_penalty=LLM_CONFIG["repetition_penalty"]
        )
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        include_images: bool = True
    ) -> Dict[str, any]:
        """
        Generate grounded answer from retrieval results
        
        Args:
            query: User question
            retrieval_results: Retrieved and reranked chunks
            include_images: Whether to include image paths in response
        
        Returns:
            Dictionary with answer, sources, images, debug info
        """
        # Build context from retrieval results
        context, sources, images = self._build_context(
            retrieval_results,
            include_images
        )
        
        # Build prompt
        prompt = self._build_prompt(query, context, sources)
        
        # Generate answer
        print("\nGenerating answer...")
        outputs = self.pipe(
            prompt,
            return_full_text=False
        )
        
        answer = outputs[0]['generated_text']
        
        # Post-process answer
        answer = self._post_process_answer(answer)
        
        # Calculate token usage
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(answer))
        
        return {
            "answer": answer,
            "sources": sources,
            "images": images if include_images else [],
            "debug": {
                "prompt": prompt,
                "num_contexts": len(retrieval_results),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
    
    def _build_context(
        self,
        results: List[RetrievalResult],
        include_images: bool
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Build context string from retrieval results
        
        Returns:
            (context_string, sources_list, image_paths)
        """
        context_parts = []
        sources = []
        images = []
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            
            # Build context entry
            context_entry = f"""
[Context {i}]
Chapter: {chunk.chapter_name} (Chapter {chunk.chapter_number})
Page: {chunk.page_number}
Type: {chunk.chunk_type}
"""
            
            # Add topic if available
            if chunk.topic:
                context_entry += f"Topic: {chunk.topic}\n"
            
            # Add content
            context_entry += f"Content: {chunk.text_content}\n"
            
            # Add caption if diagram
            if chunk.chunk_type == "diagram" and chunk.caption:
                context_entry += f"Caption: {chunk.caption}\n"
            
            context_parts.append(context_entry)
            
            # Track sources
            source = {
                "chapter_number": chunk.chapter_number,
                "chapter_name": chunk.chapter_name,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "score": result.score
            }
            sources.append(source)
            
            # Track images
            if include_images and chunk.image_path:
                images.append(chunk.image_path)
        
        context_string = "\n".join(context_parts)
        
        return context_string, sources, images
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        sources: List[Dict]
    ) -> str:
        """
        Build complete prompt for LLM
        """
        # Use template from config
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
            chapter_number="{chapter_number}",
            page_number="{page_number}"
        )
        
        # Add system prompt for models that support it
        if "Llama" in self.model_name or "Mistral" in self.model_name:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        return full_prompt
    
    def _post_process_answer(self, answer: str) -> str:
        """
        Clean up generated answer
        """
        # Remove any residual prompt artifacts
        answer = answer.strip()
        
        # Remove common artifacts
        artifacts = [
            "Answer:",
            "Assistant:",
            "AI:",
            "<|endoftext|>",
            "</s>"
        ]
        
        for artifact in artifacts:
            if answer.startswith(artifact):
                answer = answer[len(artifact):].strip()
        
        return answer
    
    def format_answer_with_citations(
        self,
        answer: str,
        sources: List[Dict]
    ) -> str:
        """
        Format answer with proper citations
        """
        # Add source citations at the end
        citation_text = "\n\n**Sources:**\n"
        
        # Group by chapter
        chapters = {}
        for source in sources:
            ch_num = source['chapter_number']
            ch_name = source['chapter_name']
            
            if ch_num not in chapters:
                chapters[ch_num] = {
                    'name': ch_name,
                    'pages': set()
                }
            chapters[ch_num]['pages'].add(source['page_number'])
        
        # Format citations
        for ch_num in sorted(chapters.keys()):
            ch_data = chapters[ch_num]
            pages = sorted(ch_data['pages'])
            
            citation_text += f"- Chapter {ch_num}: {ch_data['name']} "
            citation_text += f"(Pages: {', '.join(map(str, pages))})\n"
        
        return answer + citation_text


class PromptTemplates:
    """
    Collection of specialized prompt templates
    """
    
    @staticmethod
    def concept_explanation_template() -> str:
        return """Based on the provided NCERT Physics textbook context, explain the concept clearly.

Context:
{context}

Question: {question}

Provide a clear explanation that:
1. Defines the concept
2. Explains the underlying principles
3. Provides relevant formulas (if any)
4. References any diagrams mentioned

Answer:"""
    
    @staticmethod
    def formula_derivation_template() -> str:
        return """Based on the provided NCERT Physics textbook context, explain the formula and its derivation.

Context:
{context}

Question: {question}

Provide:
1. The formula clearly stated
2. Meaning of each variable
3. Derivation steps (if available in context)
4. Units and dimensions

Answer:"""
    
    @staticmethod
    def comparison_template() -> str:
        return """Based on the provided NCERT Physics textbook context, compare the concepts.

Context:
{context}

Question: {question}

Provide a clear comparison covering:
1. Key similarities
2. Key differences
3. When to use each
4. Examples (if available)

Answer:"""
    
    @staticmethod
    def example_solution_template() -> str:
        return """Based on the provided NCERT Physics textbook context, solve the problem step-by-step.

Context:
{context}

Question: {question}

Provide:
1. Given information
2. Required formula
3. Step-by-step solution
4. Final answer with units

Answer:"""


def select_prompt_template(query_intent: str) -> str:
    """
    Select appropriate prompt template based on query intent
    """
    templates = {
        "concept_explanation": PromptTemplates.concept_explanation_template(),
        "formula_request": PromptTemplates.formula_derivation_template(),
        "comparison": PromptTemplates.comparison_template(),
        "example_request": PromptTemplates.example_solution_template()
    }
    
    return templates.get(query_intent, ANSWER_PROMPT_TEMPLATE)


if __name__ == "__main__":
    # Example usage will be in app.py
    print("LLM module loaded. Use in main application.")