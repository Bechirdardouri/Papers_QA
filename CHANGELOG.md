# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-20

### Added

- Initial production release
- Core modules for data loading, processing, and validation
- Embedding generation using sentence-transformers (BGE, MiniLM, MPNet)
- FAISS-based vector indexing and retrieval
- LLM-based QA generation using Mistral-7B-Instruct
- Comprehensive evaluation metrics (BLEU, ROUGE, semantic similarity)
- FastAPI-based REST API with health checks
- Command-line interface with generate, index, query, evaluate, and serve commands
- Docker support with multi-stage builds
- Docker Compose configuration with monitoring stack
- GitHub Actions CI/CD pipeline
- Comprehensive test suite with pytest
- Pydantic v2 configuration management
- Structured logging with structlog and Rich
- Prometheus metrics integration

### Documentation

- README with quick start guide
- SETUP_GUIDE with complete configuration reference
- CONTRIBUTING guidelines
- API documentation via FastAPI OpenAPI

### Infrastructure

- Makefile for common development tasks
- Pre-commit hooks configuration
- Ruff and Black for code formatting
- MyPy for type checking

## [Unreleased]

### Planned

- Support for additional embedding models
- Streaming API endpoints
- WebSocket support for real-time generation
- Model fine-tuning utilities
- Distributed indexing support
- Additional evaluation metrics (BERTScore, MoverScore)
