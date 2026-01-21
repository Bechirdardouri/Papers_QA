# Papers QA: Medical Paper Question Answering System

A production-grade system for automatic question-answer pair generation, semantic retrieval, and evaluation from medical research papers. Built with state-of-the-art NLP techniques including transformer-based embeddings, FAISS vector search, and LLM-powered generation.

## Overview

Papers QA provides an end-to-end pipeline for:

- **QA Generation**: Automatically generate high-quality question-answer pairs from medical papers using Mistral-7B-Instruct with 4-bit quantization
- **Semantic Search**: Index and retrieve relevant passages using BGE embeddings and FAISS
- **Evaluation**: Comprehensive metrics including BLEU, ROUGE, and semantic similarity
- **Production Deployment**: REST API, Docker support, and monitoring integration

## Features

| Feature | Description |
|---------|-------------|
| LLM-based Generation | Mistral-7B-Instruct with 4-bit quantization for efficient QA generation |
| Vector Search | BGE-small embeddings with FAISS for fast similarity search |
| Comprehensive Metrics | BLEU, ROUGE-1, ROUGE-L, and semantic similarity evaluation |
| Environment Config | Pydantic v2-based configuration with validation |
| Structured Logging | structlog + Rich for production-grade logging |
| REST API | FastAPI-based API for production deployment |
| Docker Support | Multi-stage builds with health checks |
| Monitoring | Prometheus and Grafana integration |

## Project Structure

```
Papers_QA/
├── src/papers_qa/          # Main application package
│   ├── __init__.py         # Public API exports
│   ├── config.py           # Configuration management
│   ├── logging_config.py   # Structured logging setup
│   ├── cli.py              # Command-line interface
│   ├── api/                # REST API module
│   ├── data/               # Data loading and processing
│   ├── retrieval/          # Embedding and vector search
│   ├── llm/                # LLM inference
│   ├── generation/         # QA pair generation
│   └── evaluation/         # Metrics and evaluation
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks
│   ├── 01_production_pipeline.ipynb
│   ├── 02_qa_generation.ipynb
│   ├── 03_training.ipynb
│   └── 04_inference.ipynb
├── data/                   # Data directories
│   ├── raw/                # Input documents
│   ├── generated/          # Generated QA pairs
│   └── cache/              # Embeddings and indices
├── monitoring/             # Prometheus configuration
├── docs/                   # Documentation
├── Makefile                # Development commands
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── pyproject.toml          # Project configuration
└── requirements.txt        # Dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Papers_QA.git
cd Papers_QA

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,api,docs]"
```

### Basic Usage

```python
from papers_qa import (
    DataLoader,
    DataProcessor,
    RetrieverPipeline,
    QAGenerator,
    QAEvaluator,
)

# Load and process documents
loader = DataLoader()
documents = loader.load_documents("data/raw")

processor = DataProcessor()
texts = [processor.extract_text_from_doc(doc) for doc in documents]

# Create retrieval index
retriever = RetrieverPipeline()
retriever.index_documents(texts)

# Retrieve relevant passages
results = retriever.retrieve("What is the mechanism of action?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc[:100]}...")

# Generate QA pairs
generator = QAGenerator()
qa_pairs = generator.generate_qa_pairs(texts[0])

# Evaluate answers
evaluator = QAEvaluator()
metrics = evaluator.evaluate_answer(reference="...", hypothesis="...")
```

### CLI Usage

```bash
# Generate QA pairs from documents
papers-qa generate --input data/raw --output data/generated/qa_pairs.csv

# Create search index
papers-qa index --documents data/raw/papers.json --output data/cache/index

# Query the system
papers-qa query --index data/cache --question "What is adenomyosis?" --top-k 5

# Evaluate results
papers-qa evaluate --references refs.csv --predictions preds.csv --output results.csv

# Start API server
papers-qa serve --host 0.0.0.0 --port 8000

# Show version
papers-qa version
```

### Using Make Commands

```bash
# Show available commands
make help

# Install with development dependencies
make install-dev

# Run tests
make test

# Format and lint code
make format lint

# Start API server
make api

# Build and run Docker container
make docker-build docker-run
```

## Configuration

Configuration is managed through environment variables with Pydantic validation. Copy the example file to get started:

```bash
cp .env.example .env
```

### Key Configuration Options

```bash
# General
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Model Settings
MODEL__EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MODEL__GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.1
MODEL__ENABLE_QUANTIZATION=true

# Data Paths
DATA__INPUT_DIR=./data/raw
DATA__OUTPUT_DIR=./data/generated
DATA__CACHE_DIR=./data/cache

# Retrieval
RETRIEVAL__INDEX_TYPE=faiss_flat
RETRIEVAL__NUM_NEIGHBORS=5

# Generation
GENERATION__NUM_QUESTIONS_PER_PASSAGE=3
GENERATION__BATCH_SIZE=4
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete configuration reference.

## API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and status |
| `/query` | POST | Query documents for relevant passages |
| `/index` | POST | Index new documents |
| `/index` | DELETE | Clear the document index |
| `/evaluate` | POST | Evaluate answer quality |

### API Example

```bash
# Health check
curl http://localhost:8000/health

# Query documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the treatment?", "top_k": 5}'

# Index documents
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Document 1 text...", "Document 2 text..."]}'
```

## Docker Deployment

### Using Docker Compose

```bash
# Start all services (app + monitoring)
docker-compose up -d

# View logs
docker-compose logs -f papers-qa

# Stop services
docker-compose down
```

### Manual Docker

```bash
# Build image
docker build -t papers-qa:latest .

# Run container
docker run -d \
  --name papers-qa \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  papers-qa:latest

# Run with GPU support
docker run -d \
  --name papers-qa \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  papers-qa:latest
```

## Testing

```bash
# Run all tests with coverage
make test

# Run tests without coverage
make test-fast

# Run specific test file
pytest tests/test_retrieval.py -v

# Run with verbose output
pytest tests/ -v --tb=short
```

## Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run all checks
make check
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| torch | >=2.1.0 | Deep learning framework |
| transformers | >=4.36.0 | LLM models |
| sentence-transformers | >=3.0.0 | Embedding models |
| faiss-cpu | >=1.7.4 | Vector similarity search |
| pandas | >=2.0.0 | Data manipulation |
| pydantic | >=2.5.0 | Configuration validation |
| structlog | >=24.1.0 | Structured logging |

See [requirements.txt](requirements.txt) for the complete list.

## Troubleshooting

### Out of Memory Errors

```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
export MODEL__EMBEDDING_DEVICE=cpu
export MODEL__GENERATION_DEVICE=cpu
```

### Slow Embedding Generation

- Reduce batch size: `MODEL__EMBEDDING_BATCH_SIZE=16`
- Use a smaller model: `MODEL__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
pip install -e .
```

## Documentation

| Document | Description |
|----------|-------------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Detailed setup and configuration guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [API Documentation](http://localhost:8000/docs) | Interactive API docs (when running) |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:

- [Mistral AI](https://mistral.ai/) - Language model
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers library
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence-Transformers](https://www.sbert.net/) - Semantic embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
