"""Evaluation metrics for QA systems."""

from typing import Any

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

from papers_qa.logging_config import get_logger
from papers_qa.retrieval import EmbeddingModel

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class QAEvaluator:
    """Comprehensive QA evaluation metrics."""

    def __init__(self) -> None:
        """Initialize evaluator."""
        self.embedding_model = EmbeddingModel()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"],
            use_stemmer=True,
        )

    def compute_bleu(
        self,
        reference: str,
        hypothesis: str,
        max_n: int = 4,
    ) -> float:
        """Compute BLEU score.

        Args:
            reference: Reference answer.
            hypothesis: Generated answer.
            max_n: Maximum n-gram size.

        Returns:
            float: BLEU score (0-1).
        """
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        # Create reference and hypothesis for sentence_bleu
        weights = [1.0 / max_n] * max_n
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=weights,
            smoothing_function=SmoothingFunction().method1,
        )

        return float(score)

    def compute_rouge(self, reference: str, hypothesis: str) -> dict[str, float]:
        """Compute ROUGE scores.

        Args:
            reference: Reference answer.
            hypothesis: Generated answer.

        Returns:
            dict: ROUGE-1 and ROUGE-L F1 scores.
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            "rouge1_f1": float(scores["rouge1"].fmeasure),
            "rougeL_f1": float(scores["rougeL"].fmeasure),
        }

    def compute_semantic_similarity(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """Compute semantic similarity using embeddings.

        Args:
            reference: Reference answer.
            hypothesis: Generated answer.

        Returns:
            float: Cosine similarity (0-1).
        """
        ref_embedding = self.embedding_model.encode(reference)
        hyp_embedding = self.embedding_model.encode(hypothesis)

        similarity = cosine_similarity([ref_embedding[0]], [hyp_embedding[0]])[0][0]
        # Clamp to [0, 1] to handle floating point precision issues
        return float(min(1.0, max(0.0, similarity)))

    def evaluate_answer(
        self,
        reference: str,
        hypothesis: str,
        compute_bleu: bool = True,
        compute_rouge: bool = True,
        compute_semantic: bool = True,
    ) -> dict[str, Any]:
        """Comprehensive answer evaluation.

        Args:
            reference: Reference answer.
            hypothesis: Generated answer.
            compute_bleu: Whether to compute BLEU.
            compute_rouge: Whether to compute ROUGE.
            compute_semantic: Whether to compute semantic similarity.

        Returns:
            dict: All computed metrics.
        """
        metrics: dict[str, Any] = {}

        if compute_bleu:
            try:
                metrics["bleu"] = self.compute_bleu(reference, hypothesis)
            except Exception as e:
                logger.error("bleu_computation_failed", error=str(e))
                metrics["bleu"] = 0.0

        if compute_rouge:
            try:
                metrics.update(self.compute_rouge(reference, hypothesis))
            except Exception as e:
                logger.error("rouge_computation_failed", error=str(e))
                metrics["rouge1_f1"] = 0.0
                metrics["rougeL_f1"] = 0.0

        if compute_semantic:
            try:
                metrics["semantic_similarity"] = self.compute_semantic_similarity(
                    reference,
                    hypothesis,
                )
            except Exception as e:
                logger.error("semantic_similarity_computation_failed", error=str(e))
                metrics["semantic_similarity"] = 0.0

        # Compute overall score (average of normalized metrics)
        scores = [v for k, v in metrics.items() if isinstance(v, (int, float))]
        if scores:
            metrics["overall_score"] = float(np.mean(scores))

        return metrics

    def evaluate_retrieval(
        self,
        retrieved_docs: list[str],
        relevant_docs: list[str],
    ) -> dict[str, float]:
        """Evaluate retrieval performance.

        Args:
            retrieved_docs: Documents retrieved by system.
            relevant_docs: Ground truth relevant documents.

        Returns:
            dict: Retrieval metrics (precision, recall, F1).
        """
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        tp = len(retrieved_set & relevant_set)
        fp = len(retrieved_set - relevant_set)
        fn = len(relevant_set - retrieved_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


class BatchEvaluator:
    """Batch evaluation of QA systems."""

    def __init__(self) -> None:
        """Initialize batch evaluator."""
        self.evaluator = QAEvaluator()

    def evaluate_qa_pairs(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, Any]:
        """Evaluate multiple QA pairs.

        Args:
            references: List of reference answers.
            hypotheses: List of generated answers.

        Returns:
            dict: Aggregated metrics.
        """
        all_metrics = []

        for ref, hyp in zip(references, hypotheses, strict=False):
            metrics = self.evaluator.evaluate_answer(ref, hyp)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    aggregated[f"{key}_mean"] = float(np.mean(values))
                    aggregated[f"{key}_std"] = float(np.std(values))
                    aggregated[f"{key}_min"] = float(np.min(values))
                    aggregated[f"{key}_max"] = float(np.max(values))

        aggregated["num_pairs"] = len(all_metrics)

        logger.info("batch_evaluation_complete", metrics=aggregated)
        return aggregated
