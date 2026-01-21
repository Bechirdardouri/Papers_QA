"""Integration tests for Papers QA system.

These tests verify that different components work together correctly.
"""

import tempfile
from pathlib import Path

import pytest

from papers_qa import (
    DataLoader,
    DataProcessor,
    RetrieverPipeline,
    Settings,
    get_settings,
    set_settings,
)


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def settings(self, temp_dir):
        """Create settings for integration tests."""
        settings = Settings(
            environment="development",
            debug=False,
            data={"input_dir": temp_dir / "input", "output_dir": temp_dir / "output", "cache_dir": temp_dir / "cache"},
        )
        set_settings(settings)
        return settings

    def test_data_loading_and_processing(self, sample_json_document, temp_dir):
        """Test data loading and processing workflow."""
        import json

        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        with open(input_dir / "test_doc.json", "w") as f:
            json.dump(sample_json_document, f)

        loader = DataLoader()
        documents = loader.load_documents(input_dir)

        assert len(documents) == 1
        assert documents[0]["title"] == "A Study on Medical Treatments"

        processor = DataProcessor()
        text = processor.extract_text_from_doc(documents[0])

        assert "Medical research" in text
        assert "randomized controlled trial" in text

    def test_text_chunking_preserves_content(self, sample_documents):
        """Test that text chunking preserves all content."""
        processor = DataProcessor()

        for doc in sample_documents:
            chunks = processor.split_text(doc, chunk_size=50, overlap=10)

            reconstructed = chunks[0]
            for i in range(1, len(chunks)):
                reconstructed += chunks[i][10:]

            assert doc[:50] in reconstructed

    def test_retrieval_pipeline_consistency(self, sample_documents):
        """Test retrieval pipeline returns consistent results."""
        retriever = RetrieverPipeline(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        retriever.index_documents(sample_documents)

        query = "What is machine learning?"

        results1 = retriever.retrieve(query, k=3)
        results2 = retriever.retrieve(query, k=3)

        assert len(results1) == len(results2)

        for (doc1, score1), (doc2, score2) in zip(results1, results2, strict=False):
            assert doc1 == doc2
            assert abs(score1 - score2) < 0.001

    def test_retrieval_relevance(self, sample_documents):
        """Test that retrieval returns relevant documents."""
        retriever = RetrieverPipeline(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        retriever.index_documents(sample_documents)

        query = "How do machines understand images?"
        results = retriever.retrieve(query, k=1)

        top_doc, score = results[0]

        assert "vision" in top_doc.lower() or "visual" in top_doc.lower()
        assert score > 0.3

    def test_save_and_load_retriever(self, sample_documents, temp_dir):
        """Test saving and loading retriever state."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        retriever = RetrieverPipeline(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
        )

        retriever.index_documents(sample_documents)
        query = "neural networks"
        original_results = retriever.retrieve(query, k=2)

        retriever.save()

        new_retriever = RetrieverPipeline(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
        )
        new_retriever.load()

        loaded_results = new_retriever.retrieve(query, k=2)

        assert len(original_results) == len(loaded_results)
        for (doc1, _), (doc2, _) in zip(original_results, loaded_results, strict=False):
            assert doc1 == doc2


class TestDataProcessing:
    """Integration tests for data processing."""

    def test_clean_and_split_workflow(self):
        """Test cleaning and splitting text workflow."""
        processor = DataProcessor()

        raw_text = """
        This   is   some   messy   text   with   extra   spaces.

        It also has multiple paragraphs and some
        unusual    formatting    that   needs   cleaning.
        """

        cleaned = processor.clean_text(raw_text)
        assert "   " not in cleaned
        assert cleaned.startswith("This")

        chunks = processor.split_text(cleaned, chunk_size=50, overlap=10)
        assert len(chunks) > 1

        for chunk in chunks:
            assert len(chunk) <= 50 + 10

    def test_qa_validation_rules(self):
        """Test QA pair validation rules."""
        processor = DataProcessor()

        assert processor.validate_qa_pair(
            "What is the primary finding of this study?",
            "The study found that treatment A was significantly more effective than placebo.",
        )

        assert not processor.validate_qa_pair(
            "What?",
            "The study found that treatment A was significantly more effective.",
        )

        assert not processor.validate_qa_pair(
            "What is the primary finding?",
            "Yes",
        )


class TestEvaluationIntegration:
    """Integration tests for evaluation module."""

    def test_evaluation_workflow(self, sample_qa_pairs):
        """Test complete evaluation workflow."""
        from papers_qa import BatchEvaluator, QAEvaluator

        evaluator = QAEvaluator()

        reference = sample_qa_pairs[0]["answer"]
        hypothesis = "Machine learning is AI that learns from data automatically."

        metrics = evaluator.evaluate_answer(reference, hypothesis)

        assert "bleu" in metrics
        assert "rouge1_f1" in metrics
        assert "semantic_similarity" in metrics
        assert "overall_score" in metrics

        batch_evaluator = BatchEvaluator()

        references = [qa["answer"] for qa in sample_qa_pairs]
        hypotheses = [qa["answer"][:50] + "..." for qa in sample_qa_pairs]

        batch_metrics = batch_evaluator.evaluate_qa_pairs(references, hypotheses)

        assert batch_metrics["num_pairs"] == 3
        assert "bleu_mean" in batch_metrics
        assert "overall_score_mean" in batch_metrics


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    def test_settings_override(self, temp_dir):
        """Test settings can be overridden."""
        custom_settings = Settings(
            environment="development",
            debug=False,
            seed=123,
            data={
                "input_dir": temp_dir / "custom_input",
                "output_dir": temp_dir / "custom_output",
                "cache_dir": temp_dir / "custom_cache",
            },
        )

        set_settings(custom_settings)
        retrieved = get_settings()

        assert retrieved.seed == 123
        assert "custom_input" in str(retrieved.data.input_dir)

    def test_directory_creation(self, temp_dir):
        """Test that directories are created automatically."""
        settings = Settings(
            environment="development",
            debug=False,
            data={
                "input_dir": temp_dir / "auto_input",
                "output_dir": temp_dir / "auto_output",
                "cache_dir": temp_dir / "auto_cache",
            },
        )

        assert settings.data.input_dir.exists()
        assert settings.data.output_dir.exists()
        assert settings.data.cache_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
