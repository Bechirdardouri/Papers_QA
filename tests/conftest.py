"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment."""
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DEBUG"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["DATA__INPUT_DIR"] = str(Path(tmpdir) / "input")
        os.environ["DATA__OUTPUT_DIR"] = str(Path(tmpdir) / "output")
        os.environ["DATA__CACHE_DIR"] = str(Path(tmpdir) / "cache")

        Path(os.environ["DATA__INPUT_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["DATA__OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["DATA__CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

        yield

    for key in ["DATA__INPUT_DIR", "DATA__OUTPUT_DIR", "DATA__CACHE_DIR"]:
        os.environ.pop(key, None)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn from data without being explicitly programmed.",
        "Natural language processing deals with the interaction between "
        "computers and human language, including tasks like translation.",
        "Deep learning uses neural networks with many layers to model "
        "complex patterns in large amounts of data.",
        "Computer vision enables machines to interpret and understand "
        "visual information from the world around them.",
        "Reinforcement learning is a type of machine learning where agents "
        "learn to make decisions by receiving rewards or penalties.",
    ]


@pytest.fixture
def sample_qa_pairs():
    """Provide sample QA pairs for testing."""
    return [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "context": "Machine learning is a subset of artificial intelligence.",
        },
        {
            "question": "What does NLP deal with?",
            "answer": "NLP deals with interaction between computers and human language.",
            "context": "Natural language processing deals with computer-language interaction.",
        },
        {
            "question": "What are neural networks used for?",
            "answer": "Neural networks model complex patterns in large amounts of data.",
            "context": "Deep learning uses neural networks with many layers.",
        },
    ]


@pytest.fixture
def sample_json_document():
    """Provide a sample JSON document structure."""
    return {
        "title": "A Study on Medical Treatments",
        "abstract": "This study investigates the effectiveness of various treatments.",
        "body_text": [
            {"section": "Introduction", "text": "Medical research is crucial for healthcare."},
            {"section": "Methods", "text": "We conducted a randomized controlled trial."},
            {"section": "Results", "text": "The treatment showed significant improvement."},
            {"section": "Conclusion", "text": "The findings support the use of this treatment."},
        ],
    }
