# Context-Aware RAG Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Latest-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral-black.svg)](https://ollama.ai/)

> A production-ready, domain-agnostic retrieval-augmented generation system for document question answering. Built for privacy, performance, and accuracy.

## Project Overview

This system addresses the limitations of large language models when querying specific documents. Instead of relying on pre-trained knowledge that may be outdated or incorrect, it implements a retrieval-augmented generation pipeline that grounds every answer in your actual documents.

The architecture combines semantic search with keyword matching, conversation memory, and optimized chunking strategies to deliver accurate, cited responses across any document domain.

### Core Capabilities

**Document Processing**
- Domain-agnostic: Academic, technical, business, medical, legal, and general documents
- Multi-PDF support with fair chunk distribution across sources
- OCR integration for scanned or image-based PDFs
- Configurable chunking (600-word default with 75-word overlap)

**Retrieval & Generation**
- Hybrid retrieval: Semantic embeddings + keyword boosting
- Query expansion: 3-7 linguistic variations per query
- Dynamic similarity thresholds based on query type
- Conversation memory: Context from previous 3 Q&A exchanges

**Performance & Privacy**
- 100% local execution (no cloud APIs or data transmission)
- LRU caching: 95% speedup on repeated queries
- Citation tracking: Every answer references specific page numbers
- Processing speed: 15-45 seconds per PDF, 0.5-2s per query  

### System Architecture

The pipeline implements a six-stage process:

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

## Implementation Details

### Domain-Agnostic Architecture

The system avoids hardcoded domain logic through generic NLP patterns:

- Query Expansion: Linguistic transformations ("What is X?" → "X information", "X details")
- Keyword Extraction: Regex-based entity detection (emails, URLs, dates, numbers, measurements)
- No Hardcoded Vocabulary: Works for technical specs, contract terms, medical dosages, academic concepts, etc.

**Validation**: Tested across academic, technical, business, medical, legal, and general document types with consistent 75%+ accuracy on factual queries.

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

## Quick Start

### System Requirements
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
- [Poppler](https://github.com/oschwaldp/poppler-windows/releases/) (for pdf2image)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/ultimate144z/Context-Aware-RAG-Agent.git
cd Context-Aware-RAG-Agent
```

2. **Set up Python environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment** (Optional)

Create a `.env` file to redirect model caches to a different drive:
```ini
OLLAMA_MODELS=G:\AI_Projects\.ollama\models
TRANSFORMERS_CACHE=G:\AI_Projects\cache\transformers
HF_HOME=G:\AI_Projects\cache\huggingface
```

5. **Install LLM backend**
```bash
ollama pull mistral
```

6. **Verify installation**
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

### Usage

#### Web Interface (Recommended)
```bash
streamlit run src/interface/app_streamlit.py
```
Navigate to `http://localhost:8501` in your browser.

#### Command-Line Interface
```bash
python src/main.py
```
Follow interactive prompts for PDF processing and querying.

#### Jupyter Notebooks

Explore the `notebooks/` directory for:
- `pipeline_demo.ipynb`: Complete pipeline walkthrough with performance analysis
- `prompt_testing.ipynb`: Prompt engineering experiments and optimization

## Technology Stack

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

## Performance Metrics

### Retrieval Optimization
- Similarity Threshold: Lowered to 0.3 (from 0.5) for better factual query recall
- Top-K Retrieval: Increased to 8 chunks (from 5) for more context
- Query Expansion: Generates 3-7 variations per query using generic NLP patterns
- Hybrid Boosting: 15% boost for chunks with exact keyword matches
- Dynamic Thresholding: Lowers threshold for keyword-rich chunks

### Benchmarks
| Query Type | Accuracy |
|------------|----------|
| Contact info (emails, phones) | 80%+ |
| Factual details (dates, numbers) | 75%+ |
| Policy/procedure questions | 75%+ |
| Conceptual questions | 75%+ |

### Latency
- 10-page PDF: ~15 seconds
- 50-page PDF: ~45 seconds
- Query response: 0.5-2 seconds (cached: 0.1-0.5s)

## Additional Resources

### Documentation

- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md): Complete technical reference covering architecture, optimization, deployment, and troubleshooting
- [config/settings.yaml](config/settings.yaml): Tunable pipeline parameters
- [notebooks/](notebooks/): Jupyter demos for testing and experimentation

### Current Limitations

1. Conversation Memory Scope: Limited to last 3 Q&A pairs to avoid token overflow
2. Fixed Chunking: Word-count based (may split mid-sentence, though overlap helps)
3. Visual Elements: Charts, diagrams, and tables are converted to text (OCR) but lose formatting/structure
4. Single LLM: Currently supports only Mistral via Ollama
5. OCR Quality: pytesseract handles scanned PDFs but struggles with complex layouts, handwriting, or poor scan quality

Refer to [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for enhancement roadmap and implementation details.

## Contributing

This project is part of an academic portfolio. For issues, suggestions, or collaboration inquiries, please open an issue on GitHub.

## License

MIT License - See LICENSE file for details.

## Author & Contact

**Sarim Farooq**  
BS Artificial Intelligence, FAST NUCES Islamabad  
Specialization: RAG systems, document intelligence, retrieval optimization

**GitHub**: [ultimate144z](https://github.com/ultimate144z)  
**Email**: sarimfarooq1212@gmail.com

---

**Citation**: If you use this system in your research or projects, please reference this repository.