"""Papers QA: Medical Paper Question Answering System.

A production-grade system for automatic QA pair generation, retrieval,
and evaluation from medical research papers.

This package provides:
    - Document loading and preprocessing
    - Semantic embedding and vector search with FAISS
    - LLM-based QA pair generation
    - Comprehensive evaluation metrics (BLEU, ROUGE, semantic similarity)
    - REST API for production deployment
    - CLI for batch processing

Example:
    >>> from papers_qa import RetrieverPipeline, QAGenerator
    >>> retriever = RetrieverPipeline()
    >>> retriever.index_documents(["Document 1 text...", "Document 2 text..."])
    >>> results = retriever.retrieve("What is the main topic?", k=3)
"""

__version__ = "1.0.0"
__author__ = "Papers QA Team"
__email__ = "info@papersqa.com"

from papers_qa.config import Settings, get_settings, set_settings
from papers_qa.data import DataLoader, DataProcessor
from papers_qa.evaluation import BatchEvaluator, QAEvaluator
from papers_qa.generation import BatchQAGenerator, QAGenerator
from papers_qa.llm import InferencePipeline, LLMModel
from papers_qa.logging_config import PerformanceTracker, configure_logging, get_logger
from papers_qa.retrieval import EmbeddingModel, FAISSIndexer, RetrieverPipeline

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Configuration
    "Settings",
    "get_settings",
    "set_settings",
    # Logging
    "configure_logging",
    "get_logger",
    "PerformanceTracker",
    # Data processing
    "DataLoader",
    "DataProcessor",
    # Retrieval
    "EmbeddingModel",
    "FAISSIndexer",
    "RetrieverPipeline",
    # LLM
    "LLMModel",
    "InferencePipeline",
    # Generation
    "QAGenerator",
    "BatchQAGenerator",
    # Evaluation
    "QAEvaluator",
    "BatchEvaluator",
]
