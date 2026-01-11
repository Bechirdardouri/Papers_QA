"""QA generation module."""

import uuid

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from papers_qa.config import get_settings
from papers_qa.llm import InferencePipeline
from papers_qa.logging_config import get_logger

logger = get_logger(__name__)


class QAGenerator:
    """Generate QA pairs from documents."""

    GENERATION_PROMPT_TEMPLATE = """You are an expert at creating comprehensivequality question-answer pairs from medical research papers.

Given the following passage from a medical research paper, generate exactly {num_questions} high-quality, distinct questions that:
1. Test understanding of key concepts
2. Are specific and factual
3. Cannot be answered without reading the passage
4. Cover different aspects of the passage

For each question, also provide a concise, accurate answer based solely on the passage.

Format your response as JSON with this structure:
{{
    "qa_pairs": [
        {{"question": "...", "answer": "..."}}
    ]
}}

Passage:
{passage}

Generate the QA pairs:"""

    def __init__(self, model_name: str = "") -> None:
        """Initialize QA generator.

        Args:
            model_name: LLM model identifier.
        """
        self.inference_pipeline = InferencePipeline(model_name)
        settings = get_settings()
        self.num_questions = settings.generation.num_questions_per_passage

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_qa_pairs(self, passage: str) -> list[dict[str, str]]:
        """Generate QA pairs for a passage.

        Args:
            passage: Document passage.

        Returns:
            list: List of QA pair dictionaries.
        """
        logger.debug("generating_qa_pairs", passage_length=len(passage))

        prompt = self.GENERATION_PROMPT_TEMPLATE.format(
            num_questions=self.num_questions,
            passage=passage[:2000],  # Limit passage length
        )

        try:
            response = self.inference_pipeline.llm.generate(prompt, max_length=1024)
            qa_pairs = self._parse_response(response)
            logger.debug("qa_pairs_generated", count=len(qa_pairs))
            return qa_pairs
        except Exception as e:
            logger.error("qa_generation_failed", error=str(e))
            raise

    def _parse_response(self, response: str) -> list[dict[str, str]]:
        """Parse LLM response to extract QA pairs.

        Args:
            response: LLM response text.

        Returns:
            list: Extracted QA pairs.
        """
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            logger.warning("no_json_found_in_response")
            return []

        try:
            data = json.loads(json_match.group())
            qa_pairs = data.get("qa_pairs", [])
            return qa_pairs
        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e))
            return []

    def generate_dataset(
        self,
        documents: list[str],
        context_locations: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate QA dataset from documents.

        Args:
            documents: List of document texts.
            context_locations: Optional locations/sections.

        Returns:
            pd.DataFrame: Generated QA dataset.
        """
        qa_records = []

        for idx, doc in enumerate(documents):
            logger.info(
                "processing_document",
                index=idx,
                total=len(documents),
            )

            qa_pairs = self.generate_qa_pairs(doc)

            location = (
                context_locations[idx] if context_locations else f"section_{idx}"
            )

            for qa_pair in qa_pairs:
                record = {
                    "question_id": str(uuid.uuid4()),
                    "question": qa_pair.get("question", ""),
                    "answer": qa_pair.get("answer", ""),
                    "context": doc,
                    "context_location": location,
                    "paper": f"paper_{idx}",
                }
                qa_records.append(record)

        df = pd.DataFrame(qa_records)
        logger.info("dataset_generated", records=len(df))
        return df


class BatchQAGenerator:
    """Batch QA generation for efficiency."""

    def __init__(self, batch_size: int = 4, model_name: str = "") -> None:
        """Initialize batch generator.

        Args:
            batch_size: Batch size for processing.
            model_name: LLM model identifier.
        """
        self.batch_size = batch_size
        self.generator = QAGenerator(model_name)

    def generate_batch(
        self,
        passages: list[str],
        context_locations: list[str] | None = None,
    ) -> list[list[dict[str, str]]]:
        """Generate QA pairs for multiple passages.

        Args:
            passages: List of passages.
            context_locations: Optional locations.

        Returns:
            list: List of QA pair lists.
        """
        results = []

        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            batch_locations = (
                context_locations[i : i + self.batch_size]
                if context_locations
                else None
            )

            logger.info(
                "processing_batch",
                batch_start=i,
                batch_size=len(batch),
            )

            for passage, location in zip(
                batch,
                batch_locations or [None] * len(batch),
                strict=False,
            ):
                try:
                    qa_pairs = self.generator.generate_qa_pairs(passage)
                    results.append(qa_pairs)
                except Exception as e:
                    logger.error(
                        "batch_generation_failed",
                        location=location,
                        error=str(e),
                    )
                    results.append([])

        return results
