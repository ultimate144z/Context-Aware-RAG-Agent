"""
Streamlit Web Interface for RAG Agent
Purpose: Provide an interactive web UI for PDF processing and Q&A.

Features:
- Upload and process PDFs
- View database statistics
- Ask questions with real-time answers
- Display sources and citations
- Chat history
- Settings configuration

Usage:
    streamlit run src/interface/app_streamlit.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.utils.logger import setup_logger, get_logger
from src.main import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Context-Aware RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #262730;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #262730;
    }
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        color: #262730;
    }
    .stats-card h2 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .stats-card p {
        color: #666;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def load_pipeline():
    """Load RAG pipeline (cached)."""
    if st.session_state.pipeline is None:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            try:
                # Setup logger first
                setup_logger("RAG_Streamlit", log_dir="logs", log_to_file=True)
                # Initialize pipeline (it will create its own logger internally)
                st.session_state.pipeline = RAGPipeline(config_dir="config")
                st.success("✅ Pipeline initialized!")
                return True
            except Exception as e:
                st.error(f"❌ Error initializing pipeline: {e}")
                import traceback
                st.error(traceback.format_exc())
                return False
    return True


def sidebar():
    """Render sidebar with navigation and settings."""
    st.sidebar.markdown("# 🧠 RAG Agent")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📄 Process PDF", "💬 Ask Questions", "📊 Database Stats"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Load default from config if pipeline exists
    default_top_k = 8
    default_temp = 0.4
    if st.session_state.pipeline:
        try:
            default_top_k = st.session_state.pipeline.settings.get('top_k', 8)
            default_temp = st.session_state.pipeline.model_config['llm'].get('temperature', 0.4)
        except:
            pass
    
    top_k = st.sidebar.slider("Top-K Results", 1, 15, default_top_k, help="Number of chunks to retrieve (default from config)")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, default_temp, 0.1, help="LLM creativity (lower = more factual)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.info(
        "Context-Aware RAG Agent processes PDFs and answers questions "
        "using local embeddings and Ollama LLM. All data stays on your machine."
    )
    
    return page, top_k, temperature


def home_page():
    """Render home page."""
    st.markdown('<div class="main-header">🧠 Context-Aware RAG Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Local PDF Question-Answering System</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 Process")
        st.write("Upload PDFs and extract knowledge")
        st.write("• Text extraction")
        st.write("• OCR for scanned pages")
        st.write("• Automatic chunking")
    
    with col2:
        st.markdown("### 🔍 Search")
        st.write("Semantic search with embeddings")
        st.write("• Vector similarity")
        st.write("• Context retrieval")
        st.write("• Source tracking")
    
    with col3:
        st.markdown("### 🤖 Answer")
        st.write("Generate grounded answers")
        st.write("• Local LLM (Ollama)")
        st.write("• Citation support")
        st.write("• Factual responses")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.pipeline:
        try:
            stats = st.session_state.pipeline.chroma_manager.get_collection_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'<div class="stats-card"><h2>{stats["count"]}</h2><p>Documents in Database</p></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="stats-card"><h2>{stats["collection_name"]}</h2><p>Active Collection</p></div>', unsafe_allow_html=True)
            
            with col3:
                status = "🟢 Ready" if stats["count"] > 0 else "🟡 Empty"
                st.markdown(f'<div class="stats-card"><h2>{status}</h2><p>System Status</p></div>', unsafe_allow_html=True)
        except:
            pass


def process_pdf_page():
    """Render PDF processing page."""
    st.markdown("## 📄 Process PDF Documents")
    st.markdown("Upload a PDF to extract text, create embeddings, and add to the knowledge base.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Select a PDF document to process"
    )
    
    if uploaded_file:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        # Save uploaded file temporarily
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process PDF", type="primary"):
            st.session_state.processing = True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process PDF
                status_text.text("Step 1/6: Extracting text...")
                progress_bar.progress(0.17)
                
                success = st.session_state.pipeline.process_pdf(temp_path)
                
                progress_bar.progress(1.0)
                
                if success:
                    st.success("✅ PDF processed successfully!")
                    st.balloons()
                    
                    # Show results
                    stats = st.session_state.pipeline.chroma_manager.get_collection_stats()
                    st.info(f"📊 Total chunks in database: {stats['count']}")
                else:
                    st.error("❌ Error processing PDF. Check logs for details.")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
            
            finally:
                st.session_state.processing = False
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    st.markdown("---")
    st.markdown("### 🗂️ Batch Process Directory")
    st.write("Process all PDFs under data/raw_pdfs (recursively). Duplicates in the vector store are skipped.")
    if st.button("📥 Process All PDFs in data/raw_pdfs"):
        with st.spinner("Processing all PDFs in data/raw_pdfs..."):
            try:
                raw_dir = st.session_state.pipeline.paths_config['data']['raw_pdfs']
                summary = st.session_state.pipeline.process_directory(raw_dir)
                st.success(f"Done. Processed {summary['processed']}, Succeeded {summary['succeeded']}, Failed {summary['failed']}")
            except Exception as e:
                st.error(f"❌ Error: {e}")


def ask_questions_page(top_k, temperature):
    """Render Q&A page."""
    st.markdown("## 💬 Ask Questions")
    st.markdown("Ask questions about your processed documents and get answers with citations.")
    
    # Check if database has documents
    if st.session_state.pipeline:
        stats = st.session_state.pipeline.chroma_manager.get_collection_stats()
        
        if stats['count'] == 0:
            st.warning("⚠️ No documents in database. Please process a PDF first.")
            return
        
        st.info(f"📚 {stats['count']} chunks available for search")
    
    # Check Ollama connection
    if st.session_state.pipeline and not st.session_state.pipeline.llm.check_connection():
        st.error("❌ Ollama server is not running! Start it with: `ollama serve`")
        return
    
    # Scope selection
    scope_col1, scope_col2 = st.columns([2, 1])
    with scope_col1:
        available_pdfs = []
        try:
            available_pdfs = st.session_state.pipeline.chroma_manager.list_pdf_names()
        except Exception:
            available_pdfs = []
        selected_pdfs = st.multiselect(
            "Limit search to specific PDFs (optional)",
            options=available_pdfs,
            default=[]
        )

    # Question input
    question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is attitude in IT teams?",
        key="question_input"
    )
    
    # Show conversation context indicator
    if len(st.session_state.chat_history) > 0:
        num_context = min(3, len(st.session_state.chat_history))
        st.info(f"💬 Conversation Memory Active: Using context from last {num_context} Q&A pair(s)")
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process question
    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Update settings
                st.session_state.pipeline.retriever.top_k = top_k
                st.session_state.pipeline.llm.temperature = temperature
                
                # Prepare conversation history (last 3 Q&A pairs for context)
                # Format: [{"question": "...", "answer": "..."}, ...]
                conversation_history = []
                if len(st.session_state.chat_history) > 0:
                    # Get last 3 Q&A pairs
                    recent_chats = st.session_state.chat_history[-3:]
                    for chat in recent_chats:
                        conversation_history.append({
                            "question": chat['question'],
                            "answer": chat['result']['answer']
                        })
                
                # Get answer with conversation context
                result = st.session_state.pipeline.ask_question(
                    question, 
                    pdf_scope=selected_pdfs if selected_pdfs else None,
                    conversation_history=conversation_history if conversation_history else None
                )
                
                # Add to history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'question': question,
                    'result': result
                })
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**[{chat['timestamp']}] Question:**")
                st.markdown(f"💬 {chat['question']}")
                
                st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"🤖 **Answer:**")
                st.markdown(chat['result']['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                if chat['result'].get('sources'):
                    with st.expander("📚 View Sources"):
                        for source in chat['result']['sources']:
                            st.markdown(
                                f'<div class="source-box">'
                                f'📄 **{source["pdf_name"]}** - Pages: {source["pages"]}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                
                st.markdown("---")


def database_stats_page():
    """Render database statistics page."""
    st.markdown("## 📊 Database Statistics")
    
    if st.session_state.pipeline:
        try:
            stats = st.session_state.pipeline.chroma_manager.get_collection_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 General Stats")
                st.write(f"**Collection Name:** {stats['collection_name']}")
                st.write(f"**Total Chunks:** {stats['count']}")
                st.write(f"**Status:** {'✅ Active' if stats['exists'] else '❌ Not Found'}")
                st.write(f"**Location:** {stats['persist_directory']}")
            
            with col2:
                st.markdown("### ⚙️ Configuration")
                config = st.session_state.pipeline.settings
                st.write(f"**Chunk Size:** {config['chunk_size']} words")
                st.write(f"**Overlap:** {config['overlap']} words")
                st.write(f"**Top-K:** {config['top_k']}")
                st.write(f"**Embedding Model:** {st.session_state.pipeline.model_config['embedding_model']['name']}")
            
            st.markdown("---")
            
            # Collections list
            st.markdown("### 📚 All Collections")
            collections = st.session_state.pipeline.chroma_manager.list_all_collections()
            
            if collections:
                for col_name in collections:
                    st.write(f"• {col_name}")
            else:
                st.info("No collections found")
            
            st.markdown("---")
            
            # Danger zone
            st.markdown("### ⚠️ Danger Zone")
            
            if st.button("🗑️ Reset Database", type="secondary"):
                if st.checkbox("I understand this will delete all data"):
                    try:
                        st.session_state.pipeline.chroma_manager.reset_collection()
                        st.success("✅ Database reset successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading stats: {e}")


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Load pipeline
    if not load_pipeline():
        st.stop()
    
    # Render sidebar and get page selection
    page, top_k, temperature = sidebar()
    
    # Render selected page
    if page == "🏠 Home":
        home_page()
    elif page == "📄 Process PDF":
        process_pdf_page()
    elif page == "💬 Ask Questions":
        ask_questions_page(top_k, temperature)
    elif page == "📊 Database Stats":
        database_stats_page()


if __name__ == "__main__":
    main()
