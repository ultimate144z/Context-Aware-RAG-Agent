# Context-Aware RAG Agent

**A Domain-Agnostic Local RAG System for ANY PDF Document Type**

## Project Overview

Modern language models are powerful but often produce unreliable or hallucinated answers when asked about specific documents. This project solves that problem by building a **Retrieval-Augmented Generation (RAG) pipeline** that grounds answers in YOUR documents.

### Key Features

1. Domain-Agnostic: Works for academic papers, technical manuals, business contracts, medical documents, legal PDFs - ANY document type  
2. Multi-PDF Support: Process and query multiple PDFs simultaneously with fair result distribution  
3. Conversation Memory: Natural follow-up questions with context from previous 3 Q&A pairs  
4. Hybrid Retrieval: Combines semantic search + keyword matching for optimal accuracy  
5. Optimized Chunking: 600-word chunks (up from 400) for better context preservation  
6. 100% Local & Private: No cloud APIs, no data leakage, zero API costs  
7. OCR Support: Handles both text-based and scanned/image PDFs  
8. Citation Tracking: Every answer references specific page numbers  
9. Optimized Performance: LRU caching provides ~95% speedup on repeated queries  

### How It Works

The system processes PDFs into searchable chunks, converts them to numerical vectors (embeddings), and stores them in a local vector database. When you ask a question, it retrieves the most relevant chunks and generates an answer using a locally hosted Ollama LLM (Mistral), ensuring factual accuracy with source citations.

## Core Pipeline
```
User Uploads PDF
        ↓
Text Extraction (pdfplumber + OCR)
        ↓
Smart Chunking (600 words, 75 overlap)
        ↓
Embedding Creation (384-dim vectors)
        ↓
Vector Store (ChromaDB)
        ↓
User Query
        ↓
Hybrid Retrieval (Semantic + Keyword)
   • Query expansion (3-7 variations)
   • Top-8 most relevant chunks
   • Dynamic similarity threshold (0.3)
        ↓
LLM Answer Generation (Ollama Mistral)
   • Context + Question → Single prompt
   • Grounded answers with page citations
        ↓
Display Answer + Sources
```

## Domain-Agnostic Design

Unlike systems hardcoded for specific domains, this RAG pipeline uses **generic NLP patterns** that work universally:

- Query Expansion: Linguistic transformations ("What is X?" → "X information", "X details")
- Keyword Extraction: Regex-based entity detection (emails, URLs, dates, numbers, measurements)
- No Hardcoded Vocabulary: Works for technical specs, contract terms, medical dosages, academic concepts, etc.

Tested Across 6 Domains: Academic, Technical, Business, Medical, Legal, General

## Project Structure
```
context_aware_rag_agent/
│
├── README.md                    ← Quick start guide
├── TECHNICAL_DOCUMENTATION.md   ← Complete technical reference
├── requirements.txt             ← Python dependencies
├── .env                         ← Environment variables (optional)
│
├── config/                      ← System configurations
│   ├── paths_config.json        ← Directory paths
│   ├── model_config.json        ← LLM & embedding settings
│   └── settings.yaml            ← Pipeline parameters (chunk size, top_k, threshold)
│
├── data/                        ← Generated data (auto-created)
│   ├── raw_pdfs/                ← Uploaded PDFs
│   ├── extracted_texts/         ← Extracted text (JSON)
│   ├── processed_chunks/        ← 400-word chunks with metadata
│   ├── embeddings/              ← ChromaDB vector store
│   └── temp_images/             ← Temporary OCR images
│
├── src/                         ← Source code
│   ├── main.py                  ← Entry point (orchestrates pipeline)
│   ├── pdf_extraction/          ← Text + OCR extraction
│   ├── chunking/                ← Smart chunking logic
│   ├── embeddings/              ← Vector embedding creation
│   ├── vector_store/            ← ChromaDB management
│   ├── qa_pipeline/             ← Retrieval + LLM generation
│   ├── utils/                   ← Logging, file I/O, env utils
│   └── interface/               ← Streamlit UI
│
└── notebooks/                   ← Jupyter demos & testing
    ├── pipeline_demo.ipynb
    └── prompt_testing.ipynb
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
- [Poppler](https://github.com/oschwaldp/poppler-windows/releases/) (for pdf2image)

### Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd context_aware_rag_agent
```

2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. (Optional) Configure cache paths

Create a `.env` file to redirect model caches to a different drive:
```ini
OLLAMA_MODELS=G:\AI_Projects\.ollama\models
TRANSFORMERS_CACHE=G:\AI_Projects\cache\transformers
HF_HOME=G:\AI_Projects\cache\huggingface
```

5. Download Ollama model
```bash
ollama pull mistral
```

6. Verify setup
```bash
ollama list
# Should show: mistral
```

### First Run After Cloning

When you clone this repository for the first time, the `data/` directory structure is automatically created. However, the vector database will be empty. Here's how to get started:

**Option A: Use Sample PDFs (Quick Demo)**
1. Place PDF files in `data/raw_pdfs/`
2. Run Streamlit UI: `streamlit run src/interface/app_streamlit.py`
3. Go to "Process PDF" tab and upload your PDFs
4. Wait for processing to complete (embeddings are generated automatically)
5. Switch to "Ask Questions" tab and start querying

**Option B: Pre-Processed Database**
If you have a pre-processed `data/embeddings/` directory from a previous run, copy it into the `data/` folder. The system will use the existing vector database without re-processing.

**Important Notes:**
- The first PDF processing takes ~15-45 seconds depending on file size (longer on first run as models download)
- Subsequent queries are much faster (~0.5-2s, or ~0.1-0.5s if cached)
- All embeddings and extracted text are stored locally in `data/`

### Running the System

#### Option 1: Streamlit UI (Recommended)
```bash
streamlit run src/interface/app_streamlit.py
```
Navigate to `http://localhost:8501` in your browser.

#### Option 2: Command Line
```bash
python src/main.py
```
Follow the interactive prompts to process PDFs and ask questions.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Extraction | pdfplumber, PyPDF2 | Extract selectable text |
| OCR | pytesseract, pdf2image | Handle scanned PDFs |
| Chunking | Custom sliding window | 600-word chunks with 75-word overlap |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim semantic vectors |
| Vector Store | ChromaDB | Fast similarity search |
| LLM | Ollama (Mistral) | Local answer generation |
| UI | Streamlit | Web interface |
| Caching | LRU Cache | ~95% speedup on repeat queries |

## Performance & Optimization

### Retrieval Improvements
- Similarity Threshold: Lowered to 0.3 (from 0.5) for better factual query recall
- Top-K Retrieval: Increased to 8 chunks (from 5) for more context
- Query Expansion: Generates 3-7 variations per query using generic NLP patterns
- Hybrid Boosting: 15% boost for chunks with exact keyword matches
- Dynamic Thresholding: Lowers threshold for keyword-rich chunks

### Accuracy Results
| Query Type | Accuracy |
|------------|----------|
| Contact info (emails, phones) | 80%+ |
| Factual details (dates, numbers) | 75%+ |
| Policy/procedure questions | 75%+ |
| Conceptual questions | 75%+ |

### Processing Speed
- 10-page PDF: ~15 seconds
- 50-page PDF: ~45 seconds
- Query response: 0.5-2 seconds (cached: 0.1-0.5s)

## Documentation

- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md): Complete technical reference covering architecture, optimization, deployment, and troubleshooting
- [config/settings.yaml](config/settings.yaml): Tunable pipeline parameters
- [notebooks/](notebooks/): Jupyter demos for testing and experimentation

## Known Limitations

1. Conversation Memory Scope: Limited to last 3 Q&A pairs to avoid token overflow
2. Fixed Chunking: Word-count based (may split mid-sentence, though overlap helps)
3. Visual Elements: Charts, diagrams, and tables are converted to text (OCR) but lose formatting/structure
4. Single LLM: Currently supports only Mistral via Ollama
5. OCR Quality: pytesseract handles scanned PDFs but struggles with complex layouts, handwriting, or poor scan quality

See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for improvement suggestions.

## Author

Sarim Farooq  
BS Artificial Intelligence – FAST NUCES, Islamabad  
Focus: Context-aware systems, RAG pipelines, data extraction, and generative AI.