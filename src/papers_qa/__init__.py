"""Papers QA: Medical Paper Question Answering System.

A production-grade system for automatic QA pair generation, retrieval,
and evaluation from medical research papers.
"""

__version__ = "1.0.0"
__author__ = "Papers QA Team"
__email__ = "info@papersqa.com"

from papers_qa.config import Settings, get_settings, set_settings
from papers_qa.data import DataLoader, DataProcessor
from papers_qa.evaluation import BatchEvaluator, QAEvaluator
from papers_qa.generation import BatchQAGenerator, QAGenerator
from papers_qa.llm import InferencePipeline, LLMModel
from papers_qa.logging_config import configure_logging, get_logger
from papers_qa.retrieval import EmbeddingModel, FAISSIndexer, RetrieverPipeline

__all__ = [
    "Settings",
    "get_settings",
    "set_settings",
    "configure_logging",
    "get_logger",
    "DataLoader",
    "DataProcessor",
    "EmbeddingModel",
    "FAISSIndexer",
    "RetrieverPipeline",
    "LLMModel",
    "InferencePipeline",
    "QAGenerator",
    "BatchQAGenerator",
    "QAEvaluator",
    "BatchEvaluator",
]
