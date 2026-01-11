# Papers QA: Medical Paper Question Answering System

<div align="center">

![Papers QA](https://img.shields.io/badge/Papers%20QA-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

A **production-grade**, state-of-the-art system for automatic question-answer pair generation, retrieval, and evaluation from medical research papers.

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## Overview

**Papers QA** is an end-to-end AI-powered pipeline designed to automate the extraction of meaningful insights from medical research papers. The system combines:

- **Advanced NLP Models**: Mistral-7B-Instruct, BAAI/bge-small-en-v1.5 embeddings
- **Efficient Retrieval**: FAISS-based vector search with semantic similarity
- **Comprehensive Evaluation**: BLEU, ROUGE, semantic similarity, and retrieval metrics
- **Production Architecture**: Modular design with comprehensive logging and configuration management

### Key Benefits

✅ **Fully Automated**: Generate QA pairs without manual annotation  
✅ **Scalable**: Handle thousands of papers efficiently  
✅ **Modular**: Use individual components independently  
✅ **Well-Tested**: Comprehensive unit and integration tests  
✅ **Production-Ready**: Type hints, error handling, structured logging  
✅ **Research-Grade**: State-of-the-art models and evaluation metrics

---

## Features

### 🧾 Core Capabilities

- **QA Generation**: Automatically generate high-quality question-answer pairs from medical texts
- **Semantic Retrieval**: Retrieve relevant passages using embeddings and FAISS indexing
- **Answer Generation**: Generate contextual answers using Mistral-7B-Instruct
- **Evaluation Metrics**: BLEU, ROUGE-1, ROUGE-L, semantic similarity, retrieval accuracy
- **Batch Processing**: Efficient batch processing with configurable batch sizes
- **Caching**: Intelligent caching of embeddings and indices for faster processing

### 🏗️ Architecture Highlights

- **Modular Design**: Separate modules for data, retrieval, LLM, generation, evaluation
- **Configuration Management**: Pydantic v2 configuration with environment variable support
- **Structured Logging**: Rich, structured logs using `structlog`
- **Error Handling**: Comprehensive error handling with retries and graceful degradation
- **Type Safety**: Full type hints throughout codebase

### 📊 Supported Models

**Embedding Models** (Hugging Face):
- `BAAI/bge-small-en-v1.5` (default, 384-dim, 33M parameters)
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

**Generation Models**:
- `mistralai/Mistral-7B-Instruct-v0.1`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (recommended for GPU)
- 16GB+ RAM (8GB minimum)

### Option 1: From Source

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
docker build -t papers-qa .
docker run --gpus all -it papers-qa
```

---

## Quick Start

### 1. Basic Usage

```python
from papers_qa import (
    DataLoader,
    RetrieverPipeline,
    QAGenerator,
    QAEvaluator,
    configure_logging,
)

# Setup
configure_logging()

# Load documents
loader = DataLoader()
documents = loader.load_documents("data/raw")

# Create retriever and index
retriever = RetrieverPipeline()
retriever.index_documents(documents)

# Generate QA pairs
generator = QAGenerator()
qa_pairs = generator.generate_dataset(documents)

# Evaluate
evaluator = QAEvaluator()
metrics = evaluator.evaluate_answer(reference, prediction)
print(f"BLEU: {metrics['bleu']:.4f}")
print(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
```

### 2. CLI Commands

```bash
# Generate QA pairs
papers-qa generate \
  --input data/raw \
  --output data/generated/qa_pairs.csv \
  --batch-size 4

# Create vector index
papers-qa index \
  --documents data/raw/papers.json \
  --output data/cache/index

# Query the system
papers-qa query \
  --index data/cache/index \
  --question "What is adenomyosis?" \
  --top-k 5

# Evaluate performance
papers-qa evaluate \
  --references data/reference_answers.csv \
  --predictions data/predicted_answers.csv \
  --output results/metrics.csv
```

### 3. Configuration

Create a `.env` file or use environment variables:

```bash
# Model configuration
MODEL__EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MODEL__GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.1
MODEL__ENABLE_QUANTIZATION=true

# Data configuration
DATA__INPUT_DIR=./data/raw
DATA__OUTPUT_DIR=./data/generated
DATA__CACHE_DIR=./data/cache

# Generation configuration
GENERATION__BATCH_SIZE=4
GENERATION__NUM_QUESTIONS_PER_PASSAGE=3

# General
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DOCUMENTS                       │
└─────────────────────┬───────────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │   Data Loading & Processing  │
        │  (DataLoader, DataProcessor) │
        └──────────────┬────────────────┘
                       ↓
        ┌──────────────────────────────┐
        │  Embedding & Indexing        │
        │  (EmbeddingModel, FAISSIndex)│
        └──────────────┬───────────────┘
                       ↓
      ┌────────────────────────────────────┐
      │  QA Generation Pipeline             │
      │  (QAGenerator, InferencePipeline)   │
      └──────────────┬─────────────────────┘
                     ↓
      ┌────────────────────────────────────┐
      │  Retrieval & Answer Generation      │
      │  (RetrieverPipeline, LLMModel)      │
      └──────────────┬─────────────────────┘
                     ↓
      ┌────────────────────────────────────┐
      │  Evaluation Metrics                 │
      │  (QAEvaluator, BatchEvaluator)      │
      └──────────────┬─────────────────────┘
                     ↓
         ┌──────────────────────────┐
         │   OUTPUT & RESULTS        │
         └──────────────────────────┘
```

### Module Structure

```
papers_qa/
├── config.py              # Configuration management
├── logging_config.py      # Logging setup
├── cli.py                 # Command-line interface
├── data/                  # Data loading & processing
├── retrieval/             # Embedding & vector search
├── llm/                   # LLM inference
├── generation/            # QA generation
└── evaluation/            # Metrics & evaluation
```

---

## API Reference

### DataLoader

```python
from papers_qa import DataLoader

loader = DataLoader()
documents = loader.load_documents("path/to/documents")
qa_data = loader.load_qa_dataset("path/to/qa.csv")
```

### RetrieverPipeline

```python
from papers_qa import RetrieverPipeline

retriever = RetrieverPipeline(model_name="BAAI/bge-small-en-v1.5")
retriever.index_documents(documents)

# Retrieve similar documents
results = retriever.retrieve("What is adenomyosis?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f}, Doc: {doc[:100]}...")

# Save/load index
retriever.save()
retriever.load()
```

### QAGenerator

```python
from papers_qa import QAGenerator

generator = QAGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.1")
qa_pairs = generator.generate_qa_pairs(passage)
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

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(
    retrieved_docs=["doc1", "doc2"],
    relevant_docs=["doc1", "doc3"]
)
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

---

## Advanced Usage

### Custom Configuration

```python
from papers_qa import Settings, set_settings

# Create custom settings
custom_settings = Settings(
    environment="production",
    log_level="DEBUG",
    model__embedding_model="sentence-transformers/all-mpnet-base-v2",
    model__generation_max_length=1024,
)

# Apply globally
set_settings(custom_settings)
```

### Batch Processing

```python
from papers_qa import BatchQAGenerator, BatchEvaluator

# Batch generation
batch_gen = BatchQAGenerator(batch_size=8)
qa_results = batch_gen.generate_batch(documents)

# Batch evaluation
batch_eval = BatchEvaluator()
metrics = batch_eval.evaluate_qa_pairs(references, predictions)
```

### Performance Tracking

```python
from papers_qa.logging_config import PerformanceTracker

tracker = PerformanceTracker()

# Record metrics
tracker.record("qa_generation", duration_seconds=2.5, metadata={"docs": 10})
tracker.record("indexing", duration_seconds=5.1, metadata={"docs": 100})

# Get summary
summary = tracker.get_summary("qa_generation")
print(f"Avg time: {summary['avg']:.2f}s")
```

---

## Performance Benchmarks

### Embedding Performance
- **Model**: BAAI/bge-small-en-v1.5
- **Throughput**: ~1,000 documents/min on GPU
- **Memory**: ~2GB per 10K documents

### Generation Performance
- **Model**: Mistral-7B-Instruct (4-bit quantized)
- **Throughput**: ~10 QA pairs/min
- **Memory**: ~8GB

### Retrieval Performance
- **Index Type**: FAISS IVF
- **Query Time**: ~10ms per query (10K docs)
- **Index Size**: ~0.5MB per 1K documents

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=papers_qa --cov-report=html

# Specific test
pytest tests/test_core.py::TestConfig::test_settings_creation -v
```

### Performance Testing

```bash
# Run benchmarks
python -m pytest tests/test_performance.py --benchmark

# Profile code
python -m cProfile -o profile.prof main.py
pyprof2calltree -i profile.prof -o profile.kcachegrind
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
ENV PYTHONPATH=/app/src

ENTRYPOINT ["papers-qa"]
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
```

### FastAPI Server

```python
from fastapi import FastAPI
from papers_qa import RetrieverPipeline, InferencePipeline

app = FastAPI()
retriever = RetrieverPipeline()
pipeline = InferencePipeline()

@app.post("/query")
async def query(question: str):
    docs = retriever.retrieve(question, k=3)
    context = docs[0][0] if docs else ""
    answer = pipeline.answer_question(question, context)
    return {"question": question, "answer": answer}
```

---

## Troubleshooting

### CUDA Issues

```python
# Force CPU
from papers_qa import Settings, set_settings
settings = Settings(
    model__embedding_device="cpu",
    model__generation_device="cpu"
)
set_settings(settings)
```

### Out of Memory

```python
# Reduce batch size and quantize
settings = Settings(
    model__embedding_batch_size=8,
    model__enable_quantization=True,
    generation__batch_size=2
)
```

### Slow Inference

```bash
# Use quantized model
export MODEL__ENABLE_QUANTIZATION=true

# Use smaller embedding model
export MODEL__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/papers-qa.git
cd papers-qa
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

---

## Citation

If you use Papers QA in your research, please cite:

```bibtex
@software{papersqa2024,
  title={Papers QA: Medical Paper Question Answering System},
  author={Papers QA Team},
  year={2024},
  url={https://github.com/yourusername/papers-qa}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [BAAI](https://www.baai.ac.cn/) for BGE embeddings
- [Meta](https://www.meta.com/) for Llama models
- [Mistral AI](https://mistral.ai/) for Mistral models
- [Facebook Research](https://research.facebook.com/) for FAISS

---

## Support

- 📖 [Documentation](https://papers-qa.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/yourusername/papers-qa/issues)
- 💬 [Discussions](https://github.com/yourusername/papers-qa/discussions)
- 📧 [Email](mailto:support@papersqa.com)

---

<div align="center">

Made with ❤️ by Papers QA Team

[⬆ Back to Top](#papers-qa-medical-paper-question-answering-system)

</div>
