"""
Streamlit Application for NCERT Physics RAG System
"""

import streamlit as st
from pathlib import Path
import json
from PIL import Image
import traceback

from embeddings import EmbeddingGenerator, create_vector_store
from retrieval import RetrievalPipeline
from llm_generator import AnswerGenerator
from config import (
    UI_CONFIG,
    EMBEDDING_CONFIG,
    VECTOR_STORE_CONFIG,
    METADATA_DIR,
    LLM_CONFIG
)


# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["sidebar_state"]
)


@st.cache_resource
def load_system():
    """
    Load all system components (cached for performance)
    """
    with st.spinner("Loading RAG system..."):
        # Load embedder
        embedder = EmbeddingGenerator(
            model_name=EMBEDDING_CONFIG["model_name"]
        )
        
        # Load vector store
        vector_store = create_vector_store(
            store_type=VECTOR_STORE_CONFIG["type"],
            dimension=embedder.dimension
        )
        
        # For ChromaDB, no need to explicitly load
        # For FAISS, load the index
        if VECTOR_STORE_CONFIG["type"] == "faiss":
            vector_store.load()
        
        # Load chunks metadata for BM25
        chunks_metadata = load_chunks_metadata()
        
        # Initialize retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(
            vector_store=vector_store,
            embedder=embedder,
            chunks_metadata=chunks_metadata
        )
        
        # Initialize LLM (optional - can be loaded on demand)
        # llm = AnswerGenerator(model_name=LLM_CONFIG["model_name"])
        
        return embedder, vector_store, retrieval_pipeline


def load_chunks_metadata():
    """Load chunks metadata from JSON files"""
    chunks_metadata = []
    
    # Try to load chunks from both class 11 and 12
    for class_level in ["11", "12"]:
        chunks_file = METADATA_DIR / f"chunks_{class_level}.json"
        
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
                chunks_metadata.extend(chunks)
    
    return chunks_metadata


def display_header():
    """Display application header"""
    st.title("📚 NCERT Physics RAG System")
    st.markdown("""
    **Intelligent Physics Tutor** powered by Retrieval-Augmented Generation
    
    Ask questions about NCERT Physics Class 11 & 12 textbooks and get accurate, 
    textbook-grounded answers with relevant diagrams and formulas.
    """)
    st.divider()


def display_sidebar():
    """Display sidebar with filters and options"""
    st.sidebar.header("⚙️ Settings")
    
    # Chapter filter
    chapter_filter = st.sidebar.selectbox(
        "Filter by Chapter",
        options=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        format_func=lambda x: "All Chapters" if x is None else f"Chapter {x}"
    )
    
    # Number of results
    num_results = st.sidebar.slider(
        "Number of contexts to retrieve",
        min_value=3,
        max_value=15,
        value=5,
        help="More contexts = more comprehensive but slower"
    )
    
    # Show debug panel
    show_debug = st.sidebar.checkbox(
        "Show Debug Information",
        value=UI_CONFIG["show_debug"],
        help="Display retrieved chunks, scores, and prompt"
    )
    
    st.sidebar.divider()
    
    # Example questions
    st.sidebar.header("💡 Example Questions")
    
    example_questions = [
        "Explain Young's modulus",
        "What is the stress-strain curve?",
        "Derive the formula for elastic potential energy",
        "Show me a diagram of springs in series",
        "Compare stress and strain",
        "What is Hooke's law?",
        "Explain Poisson's ratio with examples",
        "Summarize Chapter 9 on elasticity"
    ]
    
    for i, question in enumerate(example_questions):
        if st.sidebar.button(question, key=f"example_{i}"):
            st.session_state.query = question
    
    return {
        "chapter_filter": chapter_filter,
        "num_results": num_results,
        "show_debug": show_debug
    }


def display_query_input():
    """Display query input area"""
    st.subheader("🔍 Ask Your Question")
    
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_area(
        "Enter your physics question:",
        value=st.session_state.query,
        height=100,
        placeholder="e.g., Explain the concept of elasticity and Young's modulus",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.query = ""
        st.rerun()
    
    return query, search_button


def display_results(answer_data, settings):
    """Display answer and supporting information"""
    
    # Main answer
    st.subheader("📝 Answer")
    
    answer = answer_data["answer"]
    sources = answer_data["sources"]
    images = answer_data.get("images", [])
    debug = answer_data.get("debug", {})
    
    # Display answer
    st.markdown(answer)
    
    # Display images if available
    if images:
        st.divider()
        st.subheader("📊 Related Diagrams")
        
        cols = st.columns(min(len(images), 3))
        
        for i, image_path in enumerate(images[:6]):  # Max 6 images
            with cols[i % 3]:
                try:
                    img = Image.open(image_path)
                    st.image(
                        img,
                        caption=f"Figure {i+1}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Could not load image: {image_path}")
    
    # Display sources
    st.divider()
    st.subheader("📚 Sources")
    
    # Group sources by chapter
    chapters = {}
    for source in sources:
        ch_num = source['chapter_number']
        ch_name = source['chapter_name']
        
        if ch_num not in chapters:
            chapters[ch_num] = {
                'name': ch_name,
                'pages': set(),
                'types': []
            }
        chapters[ch_num]['pages'].add(source['page_number'])
        chapters[ch_num]['types'].append(source['chunk_type'])
    
    # Display sources
    for ch_num in sorted(chapters.keys()):
        ch_data = chapters[ch_num]
        pages = sorted(ch_data['pages'])
        
        st.markdown(
            f"**Chapter {ch_num}: {ch_data['name']}** "
            f"(Pages: {', '.join(map(str, pages))})"
        )
    
    # Debug panel
    if settings["show_debug"]:
        st.divider()
        st.subheader("🔧 Debug Information")
        
        # Token usage
        if debug:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Input Tokens", debug.get('input_tokens', 'N/A'))
            with col2:
                st.metric("Output Tokens", debug.get('output_tokens', 'N/A'))
            with col3:
                st.metric("Total Tokens", debug.get('total_tokens', 'N/A'))
        
        # Retrieved contexts
        with st.expander("Retrieved Contexts", expanded=False):
            st.metric("Number of contexts", debug.get('num_contexts', 'N/A'))
            
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Context {i}** (Score: {source.get('score', 0):.3f})")
                st.markdown(f"- Type: {source['chunk_type']}")
                st.markdown(f"- Chapter {source['chapter_number']}, Page {source['page_number']}")
        
        # Full prompt
        with st.expander("Complete Prompt Sent to LLM", expanded=False):
            if 'prompt' in debug:
                st.code(debug['prompt'], language="text")


def process_query(query, settings, retrieval_pipeline):
    """Process user query and generate answer"""
    
    with st.spinner("🔍 Retrieving relevant content..."):
        # Retrieve relevant chunks
        results, intent = retrieval_pipeline.retrieve(
            query=query,
            chapter_filter=settings["chapter_filter"]
        )
        
        # Limit results
        results = results[:settings["num_results"]]
        
        if not results:
            st.warning("No relevant content found in the textbooks.")
            return None
    
    with st.spinner("🤖 Generating answer..."):
        # Load LLM (on demand to save memory)
        try:
            llm = AnswerGenerator(model_name=LLM_CONFIG["model_name"])
            
            # Generate answer
            answer_data = llm.generate_answer(
                query=query,
                retrieval_results=results,
                include_images=intent.needs_diagram
            )
            
            return answer_data
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.error("This might be due to insufficient GPU memory or model loading issues.")
            
            # Provide fallback: show retrieved contexts
            st.info("Showing retrieved contexts instead:")
            
            answer_data = {
                "answer": "**Retrieved Information:**\n\n" + "\n\n".join([
                    f"**From Chapter {r.chunk.chapter_number}, Page {r.chunk.page_number}:**\n{r.chunk.text_content}"
                    for r in results[:3]
                ]),
                "sources": [
                    {
                        "chapter_number": r.chunk.chapter_number,
                        "chapter_name": r.chunk.chapter_name,
                        "page_number": r.chunk.page_number,
                        "chunk_type": r.chunk.chunk_type,
                        "score": r.score
                    }
                    for r in results
                ],
                "images": [r.chunk.image_path for r in results if r.chunk.image_path],
                "debug": {
                    "num_contexts": len(results)
                }
            }
            
            return answer_data


def main():
    """Main application"""
    
    # Display header
    display_header()
    
    # Load system
    try:
        embedder, vector_store, retrieval_pipeline = load_system()
        
        # Get collection stats
        if hasattr(vector_store, 'get_collection_stats'):
            stats = vector_store.get_collection_stats()
            st.sidebar.success(
                f"✅ Loaded {stats['total_chunks']} chunks from vector database"
            )
    
    except Exception as e:
        st.error("❌ Failed to load RAG system!")
        st.error(f"Error: {str(e)}")
        st.info("""
        **Possible solutions:**
        1. Run `python build_index.py` first to create the index
        2. Ensure PDF files are in the correct location
        3. Check that all required packages are installed
        """)
        st.stop()
    
    # Display sidebar
    settings = display_sidebar()
    
    # Query input
    query, search_button = display_query_input()
    
    # Process query
    if search_button and query.strip():
        try:
            answer_data = process_query(query, settings, retrieval_pipeline)
            
            if answer_data:
                st.divider()
                display_results(answer_data, settings)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    elif search_button:
        st.warning("Please enter a question first.")
    
    # Footer
    st.divider()
    st.caption("Built with ❤️ using open-source tools | NCERT Physics RAG System")


if __name__ == "__main__":
    main()