"""Configuration management for Papers QA system.

This module provides centralized configuration using Pydantic v2 dataclasses,
with support for environment variable overrides and validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for LLM models."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model identifier from Hugging Face",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding model: cpu, cuda, mps",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding inference",
    )

    generation_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        description="Generation model identifier",
    )
    generation_device: str = Field(
        default="cuda",
        description="Device for generation model",
    )
    generation_max_length: int = Field(
        default=512,
        description="Maximum length for generated answers",
    )
    generation_temperature: float = Field(
        default=0.7,
        description="Temperature for generation sampling",
    )
    generation_top_p: float = Field(
        default=0.95,
        description="Top-p parameter for nucleus sampling",
    )

    enable_quantization: bool = Field(
        default=True,
        description="Enable 4-bit quantization for generation model",
    )


class DataConfig(BaseSettings):
    """Configuration for data processing."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    input_dir: Path = Field(
        default=Path("./data/raw"),
        description="Input directory for raw papers",
    )
    output_dir: Path = Field(
        default=Path("./data/generated"),
        description="Output directory for processed data",
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Cache directory for embeddings and indices",
    )

    max_doc_length: int = Field(
        default=4096,
        description="Maximum document length in tokens",
    )
    chunk_size: int = Field(
        default=512,
        description="Chunk size for document segmentation",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks",
    )


class RetrievalConfig(BaseSettings):
    """Configuration for retrieval system."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    index_type: Literal["faiss_flat", "faiss_ivf"] = Field(
        default="faiss_flat",
        description="Type of FAISS index",
    )
    num_neighbors: int = Field(
        default=5,
        description="Number of neighbors to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold for retrieval",
    )
    use_cache: bool = Field(
        default=True,
        description="Enable caching of embeddings",
    )


class GenerationConfig(BaseSettings):
    """Configuration for QA generation."""

    model_config = SettingsConfigDict(env_prefix="GENERATION_")

    batch_size: int = Field(
        default=4,
        description="Batch size for QA generation",
    )
    num_questions_per_passage: int = Field(
        default=3,
        description="Number of questions to generate per passage",
    )
    api_key: str = Field(
        default="",
        description="API key for external LLM services",
    )
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for API calls",
    )
    timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests",
    )


class EvaluationConfig(BaseSettings):
    """Configuration for evaluation metrics."""

    model_config = SettingsConfigDict(env_prefix="EVALUATION_")

    compute_bleu: bool = Field(default=True, description="Compute BLEU scores")
    compute_rouge: bool = Field(default=True, description="Compute ROUGE scores")
    compute_semantic_similarity: bool = Field(
        default=True, description="Compute semantic similarity"
    )
    compute_retrieval_accuracy: bool = Field(default=True, description="Compute retrieval accuracy")


class Settings(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # General settings
    environment: Literal["development", "staging", "production"] = Field(
        default="production",
        description="Execution environment",
    )
    debug: bool = Field(default=False, description="Debug mode")
    seed: int = Field(default=42, description="Random seed")

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Path | None = Field(
        default=None,
        description="Log file path (None for console only)",
    )

    # Subconfigurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        """Validate and create necessary directories."""
        self.data.input_dir.mkdir(parents=True, exist_ok=True)
        self.data.output_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Ensure production-safe settings."""
        if self.environment == "production":
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
        return self


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance.

    Args:
        settings: Settings instance to use globally.
    """
    global _settings
    _settings = settings
