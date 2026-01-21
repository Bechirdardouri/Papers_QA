"""Unit tests for evaluation module."""

import pytest

from papers_qa.evaluation import BatchEvaluator, QAEvaluator


class TestQAEvaluator:
    """Tests for QAEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for tests."""
        return QAEvaluator()

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator is not None
        assert evaluator.embedding_model is not None
        assert evaluator.rouge_scorer is not None

    def test_compute_bleu_identical(self, evaluator):
        """Test BLEU score for identical strings."""
        text = "The quick brown fox jumps over the lazy dog."
        score = evaluator.compute_bleu(text, text)

        assert 0 <= score <= 1
        assert score > 0.9

    def test_compute_bleu_different(self, evaluator):
        """Test BLEU score for different strings."""
        reference = "The cat sat on the mat."
        hypothesis = "A dog stood on the floor."

        score = evaluator.compute_bleu(reference, hypothesis)

        assert 0 <= score <= 1
        assert score < 0.5

    def test_compute_bleu_empty(self, evaluator):
        """Test BLEU score with empty strings."""
        score = evaluator.compute_bleu("some text", "")
        assert score == 0.0 or score < 0.1

    def test_compute_rouge_identical(self, evaluator):
        """Test ROUGE scores for identical strings."""
        text = "Medical research has made significant advances."
        scores = evaluator.compute_rouge(text, text)

        assert "rouge1_f1" in scores
        assert "rougeL_f1" in scores
        assert scores["rouge1_f1"] == 1.0
        assert scores["rougeL_f1"] == 1.0

    def test_compute_rouge_partial_overlap(self, evaluator):
        """Test ROUGE scores for partial overlap."""
        reference = "The patient received medication for pain."
        hypothesis = "The patient was given medication for relief."

        scores = evaluator.compute_rouge(reference, hypothesis)

        assert 0 < scores["rouge1_f1"] < 1
        assert 0 < scores["rougeL_f1"] < 1

    def test_compute_semantic_similarity_identical(self, evaluator):
        """Test semantic similarity for identical texts."""
        text = "Neural networks are used in deep learning."
        similarity = evaluator.compute_semantic_similarity(text, text)

        assert 0.99 <= similarity <= 1.0

    def test_compute_semantic_similarity_similar(self, evaluator):
        """Test semantic similarity for similar texts."""
        reference = "Machine learning algorithms can predict outcomes."
        hypothesis = "ML models are used for making predictions."

        similarity = evaluator.compute_semantic_similarity(reference, hypothesis)

        assert 0.5 < similarity < 1.0

    def test_compute_semantic_similarity_different(self, evaluator):
        """Test semantic similarity for different texts."""
        reference = "The weather is sunny today."
        hypothesis = "Complex algorithms solve optimization problems."

        similarity = evaluator.compute_semantic_similarity(reference, hypothesis)

        assert similarity < 0.5

    def test_evaluate_answer_complete(self, evaluator):
        """Test complete answer evaluation."""
        reference = "The study found that treatment A was more effective."
        hypothesis = "Treatment A showed greater effectiveness in the study."

        metrics = evaluator.evaluate_answer(reference, hypothesis)

        assert "bleu" in metrics
        assert "rouge1_f1" in metrics
        assert "rougeL_f1" in metrics
        assert "semantic_similarity" in metrics
        assert "overall_score" in metrics

        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, float))

    def test_evaluate_answer_selective_metrics(self, evaluator):
        """Test evaluation with selective metrics."""
        reference = "Test reference answer."
        hypothesis = "Test generated answer."

        metrics = evaluator.evaluate_answer(
            reference,
            hypothesis,
            compute_bleu=True,
            compute_rouge=False,
            compute_semantic=False,
        )

        assert "bleu" in metrics
        assert "rouge1_f1" not in metrics
        assert "semantic_similarity" not in metrics

    def test_evaluate_retrieval_perfect(self, evaluator):
        """Test retrieval evaluation with perfect match."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]

        metrics = evaluator.evaluate_retrieval(retrieved, relevant)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_evaluate_retrieval_partial(self, evaluator):
        """Test retrieval evaluation with partial match."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc1", "doc3", "doc5"]

        metrics = evaluator.evaluate_retrieval(retrieved, relevant)

        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 2 / 3
        assert 0 < metrics["f1"] < 1

    def test_evaluate_retrieval_no_match(self, evaluator):
        """Test retrieval evaluation with no match."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc3", "doc4"]

        metrics = evaluator.evaluate_retrieval(retrieved, relevant)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


class TestBatchEvaluator:
    """Tests for BatchEvaluator class."""

    @pytest.fixture
    def batch_evaluator(self):
        """Create batch evaluator for tests."""
        return BatchEvaluator()

    def test_batch_evaluator_initialization(self, batch_evaluator):
        """Test batch evaluator initializes correctly."""
        assert batch_evaluator is not None
        assert batch_evaluator.evaluator is not None

    def test_evaluate_qa_pairs_single(self, batch_evaluator):
        """Test batch evaluation with single pair."""
        references = ["The answer is correct."]
        hypotheses = ["The answer is correct."]

        metrics = batch_evaluator.evaluate_qa_pairs(references, hypotheses)

        assert "num_pairs" in metrics
        assert metrics["num_pairs"] == 1
        assert "bleu_mean" in metrics
        assert "overall_score_mean" in metrics

    def test_evaluate_qa_pairs_multiple(self, batch_evaluator):
        """Test batch evaluation with multiple pairs."""
        references = [
            "Treatment was effective.",
            "The patient recovered.",
            "Results were significant.",
        ]
        hypotheses = [
            "The treatment showed effectiveness.",
            "Patient showed recovery.",
            "Significant results were observed.",
        ]

        metrics = batch_evaluator.evaluate_qa_pairs(references, hypotheses)

        assert metrics["num_pairs"] == 3
        assert "bleu_mean" in metrics
        assert "bleu_std" in metrics
        assert "bleu_min" in metrics
        assert "bleu_max" in metrics

    def test_evaluate_qa_pairs_aggregation(self, batch_evaluator):
        """Test that aggregation statistics are calculated correctly."""
        references = ["Same text.", "Same text.", "Same text."]
        hypotheses = ["Same text.", "Same text.", "Same text."]

        metrics = batch_evaluator.evaluate_qa_pairs(references, hypotheses)

        assert metrics["bleu_std"] < 0.01
        assert metrics["bleu_min"] == metrics["bleu_max"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
