# Papers QA Setup Guide

Complete setup, configuration, and API reference for the Papers QA system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Architecture](#architecture)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- Python 3.10 or higher
- CUDA 11.8+ (recommended for GPU acceleration)
- 16GB+ RAM (8GB minimum for CPU-only mode)
- 10GB+ disk space for models and data

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| GPU | None | NVIDIA RTX 3080+ |
| Storage | 10GB | 50GB+ SSD |

---

## Installation

### Option 1: From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/yourusername/papers-qa.git
cd papers-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install optional GPU support
pip install ".[gpu]"
```

### Option 2: Direct Installation

```bash
pip install papers-qa
```

### Option 3: Docker

```bash
# Build image
docker build -t papers-qa .

# Run with GPU
docker run --gpus all -it papers-qa

# Run CPU-only
docker run -it papers-qa
```

### Verify Installation

```python
from papers_qa import __version__, get_settings
print(f"Papers QA v{__version__}")
settings = get_settings()
print(f"Environment: {settings.environment}")
```

---

## Configuration

### Environment Variables

Configuration is managed through environment variables with Pydantic validation. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### Configuration Reference

#### General Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | string | production | Execution environment (development, staging, production) |
| `DEBUG` | bool | false | Enable debug mode (disabled in production) |
| `SEED` | int | 42 | Random seed for reproducibility |
| `LOG_LEVEL` | string | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | path | None | Log file path (None for console only) |

#### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL__EMBEDDING_MODEL` | string | BAAI/bge-small-en-v1.5 | Embedding model identifier |
| `MODEL__EMBEDDING_DEVICE` | string | cpu | Device for embeddings (cpu, cuda, mps) |
| `MODEL__EMBEDDING_BATCH_SIZE` | int | 32 | Batch size for embedding generation |
| `MODEL__GENERATION_MODEL` | string | mistralai/Mistral-7B-Instruct-v0.1 | LLM model identifier |
| `MODEL__GENERATION_DEVICE` | string | cuda | Device for generation |
| `MODEL__GENERATION_MAX_LENGTH` | int | 512 | Maximum generation length |
| `MODEL__GENERATION_TEMPERATURE` | float | 0.7 | Sampling temperature |
| `MODEL__GENERATION_TOP_P` | float | 0.95 | Nucleus sampling parameter |
| `MODEL__ENABLE_QUANTIZATION` | bool | true | Enable 4-bit quantization |

#### Data Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA__INPUT_DIR` | path | ./data/raw | Input documents directory |
| `DATA__OUTPUT_DIR` | path | ./data/generated | Output directory for results |
| `DATA__CACHE_DIR` | path | ./data/cache | Cache directory for indices |
| `DATA__MAX_DOC_LENGTH` | int | 4096 | Maximum document length |
| `DATA__CHUNK_SIZE` | int | 512 | Text chunk size |
| `DATA__CHUNK_OVERLAP` | int | 50 | Overlap between chunks |

#### Retrieval Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RETRIEVAL__INDEX_TYPE` | string | faiss_flat | Index type (faiss_flat, faiss_ivf) |
| `RETRIEVAL__NUM_NEIGHBORS` | int | 5 | Number of neighbors to retrieve |
| `RETRIEVAL__SIMILARITY_THRESHOLD` | float | 0.5 | Minimum similarity threshold |
| `RETRIEVAL__USE_CACHE` | bool | true | Enable embedding caching |

#### Generation Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENERATION__BATCH_SIZE` | int | 4 | Batch size for QA generation |
| `GENERATION__NUM_QUESTIONS_PER_PASSAGE` | int | 3 | Questions per passage |
| `GENERATION__MAX_RETRIES` | int | 3 | Maximum retry attempts |
| `GENERATION__TIMEOUT` | int | 30 | API timeout in seconds |

### Supported Models

#### Embedding Models

| Model | Dimensions | Parameters | Notes |
|-------|------------|------------|-------|
| BAAI/bge-small-en-v1.5 | 384 | 33M | Default, good balance |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 22M | Fast, lightweight |
| sentence-transformers/all-mpnet-base-v2 | 768 | 109M | Higher quality |

#### Generation Models

| Model | Parameters | Notes |
|-------|------------|-------|
| mistralai/Mistral-7B-Instruct-v0.1 | 7B | Default, efficient |
| meta-llama/Llama-2-7b-chat-hf | 7B | Requires license |
| meta-llama/Llama-2-13b-chat-hf | 13B | Higher quality |

---

## Architecture

### System Design

```
Input Documents
       |
       v
+-------------------+
| Data Loading      |  DataLoader, DataProcessor
+-------------------+
       |
       v
+-------------------+
| Embedding         |  EmbeddingModel, FAISSIndexer
+-------------------+
       |
       v
+-------------------+
| QA Generation     |  QAGenerator, InferencePipeline
+-------------------+
       |
       v
+-------------------+
| Retrieval         |  RetrieverPipeline
+-------------------+
       |
       v
+-------------------+
| Evaluation        |  QAEvaluator, BatchEvaluator
+-------------------+
       |
       v
Output Results
```

### Module Structure

```
src/papers_qa/
├── __init__.py           # Public API exports
├── config.py             # Configuration management
├── logging_config.py     # Structured logging
├── cli.py                # Command-line interface
├── api/                  # REST API module
│   └── __init__.py
├── data/                 # Data loading and processing
│   └── __init__.py
├── retrieval/            # Embedding and vector search
│   └── __init__.py
├── llm/                  # LLM inference
│   └── __init__.py
├── generation/           # QA pair generation
│   └── __init__.py
└── evaluation/           # Metrics and evaluation
    └── __init__.py
```

---

## API Reference

### DataLoader

```python
from papers_qa import DataLoader

loader = DataLoader()

# Load documents from directory
documents = loader.load_documents("path/to/documents")

# Load QA dataset from CSV
qa_data = loader.load_qa_dataset("path/to/qa.csv")
```

### DataProcessor

```python
from papers_qa import DataProcessor

processor = DataProcessor()

# Clean text
cleaned = processor.clean_text("  messy   text  ")

# Extract text from document
text = processor.extract_text_from_doc({"title": "...", "body_text": [...]})

# Split into chunks
chunks = processor.split_text(text, chunk_size=512, overlap=50)

# Validate QA pair
is_valid = processor.validate_qa_pair(question, answer)
```

### RetrieverPipeline

```python
from papers_qa import RetrieverPipeline

retriever = RetrieverPipeline(model_name="BAAI/bge-small-en-v1.5")

# Index documents
retriever.index_documents(documents)

# Retrieve similar documents
results = retriever.retrieve("What is adenomyosis?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f}, Doc: {doc[:100]}...")

# Save and load index
retriever.save()
retriever.load()
```

### QAGenerator

```python
from papers_qa import QAGenerator

generator = QAGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.1")

# Generate QA pairs from passage
qa_pairs = generator.generate_qa_pairs(passage)

# Generate dataset from multiple documents
dataset = generator.generate_dataset(documents)
```

### QAEvaluator

```python
from papers_qa import QAEvaluator

evaluator = QAEvaluator()

# Evaluate single answer
metrics = evaluator.evaluate_answer(
    reference="The answer is neural networks",
    hypothesis="Neural networks are deep learning models"
)
# Returns: bleu, rouge1_f1, rougeL_f1, semantic_similarity, overall_score

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    retrieved_docs=["doc1", "doc2"],
    relevant_docs=["doc1", "doc3"]
)
# Returns: precision, recall, f1
```

### InferencePipeline

```python
from papers_qa import InferencePipeline

pipeline = InferencePipeline()

# Standard generation
answer = pipeline.answer_question(
    question="What causes adenomyosis?",
    context="Adenomyosis is caused by..."
)

# Streaming generation
for chunk in pipeline.answer_question_streaming(question, context):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
from papers_qa import BatchQAGenerator, BatchEvaluator

# Batch QA generation
batch_gen = BatchQAGenerator(batch_size=8)
qa_results = batch_gen.generate_batch(documents)

# Batch evaluation
batch_eval = BatchEvaluator()
metrics = batch_eval.evaluate_qa_pairs(references, predictions)
```

---

## Advanced Usage

### Custom Configuration

```python
from papers_qa import Settings, set_settings

custom_settings = Settings(
    environment="production",
    log_level="DEBUG",
    model={"embedding_model": "sentence-transformers/all-mpnet-base-v2"},
    data={"chunk_size": 1024},
)

set_settings(custom_settings)
```

### Performance Tracking

```python
from papers_qa import PerformanceTracker

tracker = PerformanceTracker()

# Record metrics
tracker.record("qa_generation", duration_seconds=2.5, metadata={"docs": 10})
tracker.record("indexing", duration_seconds=5.1, metadata={"docs": 100})

# Get summary
summary = tracker.get_summary("qa_generation")
print(f"Avg time: {summary['avg']:.2f}s")
```

### REST API Integration

```python
# Start the API server
from papers_qa.api import run_server
run_server(host="0.0.0.0", port=8000)

# Or via CLI
# papers-qa serve --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health` - Health check
- `POST /query` - Query documents
- `POST /index` - Index documents
- `DELETE /index` - Clear index
- `POST /evaluate` - Evaluate answers

---

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  papers-qa:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
      - MODEL__GENERATION_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: papers-qa
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: papers-qa
        image: papers-qa:1.0.0
        resources:
          limits:
            memory: "16Gi"
            nvidia.com/gpu: "1"
        env:
        - name: ENVIRONMENT
          value: "production"
```

---

## Troubleshooting

### CUDA Issues

```python
# Force CPU mode
from papers_qa import Settings, set_settings

settings = Settings(
    model={
        "embedding_device": "cpu",
        "generation_device": "cpu"
    }
)
set_settings(settings)
```

Or via environment variables:

```bash
export MODEL__EMBEDDING_DEVICE=cpu
export MODEL__GENERATION_DEVICE=cpu
```

### Out of Memory

Reduce batch sizes and enable quantization:

```bash
export MODEL__EMBEDDING_BATCH_SIZE=8
export MODEL__ENABLE_QUANTIZATION=true
export GENERATION__BATCH_SIZE=2
```

### Slow Inference

Use smaller models:

```bash
export MODEL__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export MODEL__ENABLE_QUANTIZATION=true
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
pip install -e ".[dev,api]"
```

### Index Loading Errors

```bash
# Clear cache and rebuild
rm -rf data/cache/*
papers-qa index --documents data/raw/papers.json --output data/cache/index
```

---

## Performance Benchmarks

| Operation | Hardware | Throughput | Memory |
|-----------|----------|------------|--------|
| Embedding (BGE-small) | RTX 3080 | ~1000 docs/min | ~2GB |
| Embedding (BGE-small) | CPU | ~100 docs/min | ~1GB |
| Generation (Mistral-7B) | RTX 3080 | ~10 QA/min | ~8GB |
| Retrieval (10K docs) | Any | ~100 queries/sec | ~500MB |

---

## Support

- Documentation: This guide and README.md
- Issues: GitHub issue tracker
- Discussions: GitHub discussions
- Email: support@papersqa.com
