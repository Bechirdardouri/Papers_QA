"""Unit tests for Papers QA system."""

import tempfile
from pathlib import Path

import pytest

from papers_qa import (
    DataProcessor,
    Settings,
    get_settings,
    set_settings,
)


class TestConfig:
    """Test configuration module."""

    def test_settings_creation(self) -> None:
        """Test settings instance creation."""
        settings = Settings()
        assert settings is not None
        assert settings.environment == "production"
        assert settings.seed == 42

    def test_settings_paths_creation(self) -> None:
        """Test that required directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            tmppath = Path(tmpdir)
            os.environ["DATA__INPUT_DIR"] = str(tmppath / "input")
            os.environ["DATA__OUTPUT_DIR"] = str(tmppath / "output")
            os.environ["DATA__CACHE_DIR"] = str(tmppath / "cache")

            settings = Settings(_env_file=None)
            assert settings.data.input_dir.exists()
            assert settings.data.output_dir.exists()
            assert settings.data.cache_dir.exists()

            # Clean up environment
            os.environ.pop("DATA__INPUT_DIR", None)
            os.environ.pop("DATA__OUTPUT_DIR", None)
            os.environ.pop("DATA__CACHE_DIR", None)

    def test_global_settings(self) -> None:
        """Test global settings getter and setter."""
        settings = Settings()
        set_settings(settings)
        retrieved = get_settings()
        assert retrieved is settings


class TestDataProcessor:
    """Test data processing utilities."""

    def test_clean_text(self) -> None:
        """Test text cleaning."""
        text = "This  is   a   test   with   extra   spaces"
        cleaned = DataProcessor.clean_text(text)
        assert cleaned == "This is a test with extra spaces"

    def test_clean_text_empty(self) -> None:
        """Test cleaning empty text."""
        assert DataProcessor.clean_text("") == ""
        assert DataProcessor.clean_text("   ") == ""

    def test_clean_text_non_string(self) -> None:
        """Test cleaning non-string input."""
        assert DataProcessor.clean_text(None) == ""
        assert DataProcessor.clean_text(123) == ""

    def test_extract_text_from_doc(self) -> None:
        """Test text extraction from document."""
        doc = {
            "title": "Test Title",
            "text": "Test content",
            "abstract": "Test abstract",
        }
        text = DataProcessor.extract_text_from_doc(doc)
        assert "Test Title" in text
        assert "Test content" in text
        assert "Test abstract" in text

    def test_extract_text_nested(self) -> None:
        """Test extraction from nested structure."""
        doc = {
            "body_text": [
                {"text": "Section 1"},
                {"text": "Section 2"},
            ]
        }
        text = DataProcessor.extract_text_from_doc(doc)
        assert "Section 1" in text
        assert "Section 2" in text

    def test_split_text_basic(self) -> None:
        """Test basic text splitting."""
        text = "a" * 1000
        chunks = DataProcessor.split_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_split_text_empty(self) -> None:
        """Test splitting empty text."""
        chunks = DataProcessor.split_text("")
        assert chunks == []

    def test_validate_qa_pair_valid(self) -> None:
        """Test validation of valid QA pair."""
        question = "What is the main topic?"
        answer = "The main topic is about neural networks and deep learning."
        assert DataProcessor.validate_qa_pair(question, answer)

    def test_validate_qa_pair_short_question(self) -> None:
        """Test validation of short question."""
        question = "What?"
        answer = "A valid answer with multiple words."
        assert not DataProcessor.validate_qa_pair(question, answer)

    def test_validate_qa_pair_short_answer(self) -> None:
        """Test validation of short answer."""
        question = "What is this?"
        answer = "Yes"
        assert not DataProcessor.validate_qa_pair(question, answer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
