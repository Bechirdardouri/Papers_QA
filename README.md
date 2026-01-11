# 📚 Papers_QA: Medical Paper Question Answering

Papers_QA builds end-to-end question answering over medical papers: generate Q&A pairs, index them with embeddings, retrieve relevant passages, and evaluate answers. It stays lightweight and practical for real-world use.

---

## 🎯 What It Does

- Generate question–answer pairs from papers (Mistral-7B-Instruct)
- Embed and index documents (BGE-small + FAISS)
- Retrieve relevant passages for any query
- Evaluate answers with BLEU, ROUGE, and semantic similarity

---

## ✨ Key Features

- LLM-based Q&A generation (Mistral-7B-Instruct, 4-bit)
- Vector search with BGE-small embeddings + FAISS
- Metrics: BLEU, ROUGE, semantic similarity
- Environment-driven config (Pydantic v2)
- Structured logging with structlog + Rich
- CLI plus Docker support for easy runs

---

## 📊 Repository Structure

- Code: [src/papers_qa](src/papers_qa) (pipeline, retrieval, generation, evaluation)
- Tests: [tests](tests) (unit tests)
- Notebooks: [notebooks](notebooks) (demos and fine-tuning)
- Data: [data](data) (raw, generated, cache)
- Docs: [README.md](README.md), [SETUP_GUIDE.md](SETUP_GUIDE.md), [CONTRIBUTING.md](CONTRIBUTING.md)
- Config: [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt), [.env.example](.env.example), [.pre-commit-config.yaml](.pre-commit-config.yaml)
- Docker: [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Papers_QA.git
cd Papers_QA

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from papers_qa import get_settings, DataProcessor, RetrieverPipeline, QAGenerator

# Load configuration
settings = get_settings()

# Process documents
processor = DataProcessor()
documents = processor.load_from_csv("data.csv")

# Create retrieval system
retriever = RetrieverPipeline()
retriever.index_documents(documents)

# Generate QA pairs
generator = QAGenerator()
qa_pairs = generator.generate_qa_pairs(documents[0])

# Retrieve and answer questions
relevant_docs = retriever.search("What is the mechanism of action?", k=3)
```

### CLI Usage

```bash
# Generate QA pairs
python -m papers_qa.cli generate --input data.csv --output qa_pairs.json

# Create search index
python -m papers_qa.cli index --documents data.csv --output index.faiss

# Query the system
python -m papers_qa.cli query "Your question here" --top-k 3

# Evaluate results
python -m papers_qa.cli evaluate --references refs.csv --predictions preds.csv
```

---

## 🔧 Configuration

The system uses environment variables for configuration. Copy the example file to get started:

```bash
cp .env.example .env
```

Configure with environment variables:

```bash
export MODEL__GENERATION_MODEL="mistralai/Mistral-7B-Instruct-v0.1"
export MODEL__EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export DATA__INPUT_DIR="./data/raw"
export DATA__OUTPUT_DIR="./data/generated"
export DATA__CACHE_DIR="./data/cache"
```

All configuration is validated with Pydantic. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete details.

---

## 🧪 Testing & Quality

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

---

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f papers-qa

# Stop all services
docker-compose down
```

### Manual Docker

```bash
# Build image
docker build -t papers-qa:latest .

# Run container
docker run -v $(pwd)/data:/app/data papers-qa:latest
```

---

## 📦 Core Modules

- `config.py` – configuration
- `logging_config.py` – logging setup
- `cli.py` – command-line entry points
- `data/` – loading and processing
- `retrieval/` – embeddings and FAISS search
- `llm/` – model inference
- `generation/` – QA generation
- `evaluation/` – metrics

---

## 📚 Dependencies

**Core Libraries:**
- `torch>=2.1.0` - Deep learning framework
- `transformers>=4.36.0` - LLM models
- `sentence-transformers>=3.0.0` - Embedding models
- `faiss-cpu>=1.7.4` - Vector similarity search
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `pydantic>=2.5.0` - Data validation
- `structlog>=24.1.0` - Structured logging
- `rich>=13.7.0` - Rich terminal output
- `pytest>=7.4.0` - Testing framework

See [requirements.txt](requirements.txt) for the complete dependency list with exact versions.


## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Complete setup, configuration, and API reference |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines and development setup |

---

## 🏆 Code Quality Metrics

| Metric | Status |
|--------|--------|
| **Linting (Ruff)** | ✅ All checks passed |
| **Type Safety** | ✅ Complete type hints |
| **Unused Code** | ✅ Zero (all components used) |
**Made for medical research and open science**
---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (Black, Ruff, type hints)
- How to write and run tests
- Pull request process
- Development environment setup

Quick start for developers:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Make your changes and run tests
pytest tests/ -v
```

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
```

**Slow Embedding Generation**
- Use fewer documents or increase batch size
- Check available CPU/GPU resources
- Reduce the number of workers

**Missing Dependencies**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
pip install -e .
```

For more help, see [SETUP_GUIDE.md](SETUP_GUIDE.md) or open an issue.

---

## 📞 Support & Help

- Docs: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Issues: open a ticket on GitHub

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.


## 🙏 Acknowledgments

Built with cutting-edge open-source technology:
- [Mistral AI](https://mistral.ai/) - High-quality language model
- [Hugging Face](https://huggingface.co/) - Model hosting and libraries
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient vector search
- [Sentence-Transformers](https://www.sbert.net/) - Semantic embeddings
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [PyTest](https://pytest.org/) - Testing framework

